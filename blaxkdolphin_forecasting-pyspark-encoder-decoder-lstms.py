!pip -q install pyspark

!pip -q install petastorm
import os

import time

import math

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession, Window

from pyspark.sql.types import *

import pyspark.sql.functions as f



from pyspark.ml import Transformer, Pipeline

from pyspark.ml.feature import VectorAssembler, OneHotEncoder



from petastorm.spark import SparkDatasetConverter, make_spark_converter

from petastorm import TransformSpec



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader



def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)



def smape(forecast, actual):

    f = np.asarray(forecast)

    a = np.asarray(actual)

    up = np.abs(f - a)

    down = (np.abs(f) + np.abs(a))/2

    np.mean(up/down)

    return np.mean(up/down)*100



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



n_store = 10

n_item = 50

n_ts = 1826

n_pred = 90
spark = SparkSession.builder.master("local[*]").appName("retail_demand_forecasting").getOrCreate()
schema = StructType([StructField("date", DateType()),StructField("store", IntegerType()),

                     StructField("item", IntegerType()),StructField("sales", FloatType())])

train_df = spark.read.csv(path = '/kaggle/input/demand-forecasting-kernels-only/train.csv', schema=schema, header = True).cache()

train_df.printSchema()



schema = StructType([StructField("id", IntegerType()),

                     StructField("date", DateType()),StructField("store", IntegerType()),

                     StructField("item", IntegerType())])

test_df = spark.read.csv(path = '/kaggle/input/demand-forecasting-kernels-only/test.csv', schema=schema, header = True).cache()

test_df.printSchema()





train_df = train_df.withColumn('type',f.lit("train"))

train_df = train_df.withColumn('id',f.lit(None))



test_df = test_df.withColumn('type',f.lit("test"))

test_df = test_df.withColumn('sales',f.lit(None))



df = train_df.unionByName(test_df)
class logTransform(Transformer):

    def __init__(self, inputCol, outputCol):

        self.inputCol = inputCol

        self.outputCol = outputCol



    def _transform(self, df):

        return df.withColumn(self.outputCol, f.log1p(f.col(self.inputCol)))

    

class targetMaker(Transformer):

    def __init__(self, inputCol, outputCol='target', dateCol='date', idCol=['store', 'item'], Range = 90):

        self.inputCol = inputCol

        self.outputCol = outputCol

        self.dateCol = dateCol

        self.idCol = idCol

        self.Range = Range

        

    def _transform(self, df):

        w = Window.partitionBy(self.idCol).orderBy(self.dateCol).rowsBetween(0, self.Range - 1)

        df = df.withColumn(self.outputCol, f.collect_list(self.inputCol).over(w))

        return df

    

class seriesMaker(Transformer):

    def __init__(self, inputCol, outputCol='input', dateCol='date', idCol=['store', 'item'], Range = 120):

        self.inputCol = inputCol

        self.outputCol = outputCol

        self.dateCol = dateCol

        self.idCol = idCol

        self.Range = Range

        

    def _transform(self, df):

        w = Window.partitionBy(self.idCol).orderBy(self.dateCol).rowsBetween(-self.Range, -1)

        df = df.withColumn(self.outputCol, f.collect_list(self.inputCol).over(w))

        return df
WINDOW_SIZE = 180

# Feature extraction

logt = logTransform(inputCol ='sales', outputCol='logSales')

tgtm = targetMaker(inputCol = 'logSales', Range = n_pred)

srsm = seriesMaker(inputCol = 'logSales', Range = WINDOW_SIZE)

encoder = OneHotEncoder(inputCols=["store","item",],outputCols=["storeVec","itemVec"])

assembler = VectorAssembler(inputCols=["storeVec","itemVec"], outputCol="covariates")



pipeline = Pipeline(stages=[logt, tgtm, srsm, encoder, assembler])

processing = pipeline.fit(df)

transformed = processing.transform(df)

transformed.printSchema()



transformed_train = transformed.filter( (f.size('input')>=WINDOW_SIZE) & (f.size('target')>=n_pred) & (f.col('type') == 'train') )

transformed_test = transformed.filter( (f.size('input')>=WINDOW_SIZE) & (f.col('type') == 'test'))
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")



converter_train = make_spark_converter(transformed_train.select('input','covariates','target'))

converter_test = make_spark_converter(transformed_test.select('input','covariates', 'store', 'item'))



print(f"train: {len(converter_train)}, val: {len(converter_test)}")
class multiStepModel(nn.Module):

    def __init__(self,  input_size = 1, covar_size = 60, output_size = 1, output_seq_len = 90, h1_dim = 32, h2_dim = 32):

        super().__init__()

        

        self.input_size = input_size

        self.output_size = output_size

        self.output_seq_len = output_seq_len

        self.h1_dim = h1_dim

        self.h2_dim = h2_dim

        

        self.lstm_layer1 = nn.LSTM(input_size + covar_size, h1_dim, num_layers = 1, batch_first = True)

        self.lstm_layer2 = nn.LSTM(h1_dim + covar_size, h2_dim, num_layers = 1, batch_first = True)

        self.fc_layer = nn.Linear(h2_dim, output_size)

        self.dropout = nn.Dropout(0.2)

        



    def forward(self, input, covar):

        input_seq_len = input.size(1)

        x = covar.unsqueeze(1).repeat(1, input_seq_len, 1) # expand to input seq length

        lstm1_input = torch.cat([input, x], dim = 2) # combine with input seq

        output, (hn, cn) = self.lstm_layer1(lstm1_input)

        lstm1_output = F.relu(hn[-1]) # get the last hidden state of the last LSTM layer

        

        x = torch.cat([lstm1_output, covar], dim = 1).unsqueeze(1) # combine with covariates

        lstm2_input = x.repeat(1, self.output_seq_len, 1)

        output, (hn, cn) = self.lstm_layer2(lstm2_input)

        lstm2_output = F.relu(output)

        

        fc_input = self.dropout(lstm2_output)

        out = self.fc_layer(lstm2_output).squeeze()

        

        return out
def train_one_epoch(dataiter, steps_per_epoch):

    model.train()  # Set model to training mode



    curr_loss = []

    for step in range(1, steps_per_epoch+1):

        pd_batch = next(dataiter)

        x1 = pd_batch['input'].unsqueeze(2).to(device)

        x2 = pd_batch['covariates'].to(device)

        y = pd_batch['target'].to(device)

    

        # Track history in training

        with torch.set_grad_enabled(True):

            optimizer.zero_grad()



            out = model(x1, x2)

            loss = loss_fn(out, y)



            loss.backward()

            optimizer.step()



        curr_loss.append(loss.item())

        print('\rprogress {:6.2f} %\tloss {:8.4f}'.format(round(100*step/steps_per_epoch, 2), np.mean(curr_loss)), end = "")

  

    epoch_loss = np.mean(curr_loss)

    print('\rprogress {:6.2f} %\tloss {:8.4f}'.format(round(100*step/steps_per_epoch, 2), epoch_loss ))

    return epoch_loss
LR = 1e-3

BATCH_SIZE = 128

NUM_EPOCHS = 50



model = multiStepModel().to(device)

optimizer = optim.Adam(model.parameters(), lr = LR)

loss_fn = nn.MSELoss()



start = time.time()

with converter_train.make_torch_dataloader(batch_size = BATCH_SIZE) as trainloader:

    trainiter = iter(trainloader)

    steps_per_epoch = len(converter_train) // BATCH_SIZE

    for epoch in range(1, NUM_EPOCHS+1):

        print('-' * 10)

        print('Epoch {}/{}\t{} batches'.format(epoch, NUM_EPOCHS, steps_per_epoch))

        epoch_loss = train_one_epoch(trainiter, steps_per_epoch)

        print('{}'.format(timeSince(start)))
date_df = test_df.select('date').distinct().sort('date').toPandas()
def forecast(model, converter_test):

    

    model.eval()  # Set model to evaluate mode

    

    with converter_test.make_torch_dataloader(batch_size = 1, num_epochs = 1) as testloader:

        testiter = iter(testloader)

        steps_per_epoch = len(converter_test)



        final_df = pd.DataFrame()



        for step in range(1, steps_per_epoch+1):

        

            pd_batch = next(testiter)

            x1 = pd_batch['input'].unsqueeze(2).to(device)

            x2 = pd_batch['covariates'].to(device)

    

            with torch.set_grad_enabled(False):

                out = model(x1, x2)

        

            curr_df = date_df

            curr_df['sales'] = np.expm1(out.tolist()).round()

            curr_df['store'] = pd_batch['store'].item()

            curr_df['item'] = pd_batch['item'].item()

            final_df = final_df.append(curr_df)

        

            print('\rprogress {:6.2f} %'.format(round(100*step/steps_per_epoch, 2)), end = "")

        print('\rprogress {:6.2f} %'.format(round(100*step/steps_per_epoch, 2)))

    

    return final_df
final_df = forecast(model, converter_test)



test_df = test_df.drop('sales')

sub_df = test_df.toPandas().merge(final_df, how = 'left', on = ['date','store','item'])



sub_df[['id','sales']].to_csv('submission.csv', index=False)