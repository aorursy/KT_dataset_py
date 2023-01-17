import pandas as pd

import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle

import torch

from torch import nn

from torch.utils.data import Dataset

import torch.nn.functional as F

from torch.autograd import Variable

from sklearn.metrics import mean_squared_error

from datetime import datetime

from datetime import timedelta

from sklearn import preprocessing

train = pd.read_csv('/kaggle/input/covid19-with-population/update_train_processed.csv')
# The character ' will make later query function report an error,so it's replaced by a space

train.Country.replace('Cote d\'Ivoire','Cote d Ivoire',inplace=True)

train.Province.replace('Cote d\'Ivoire','Cote d Ivoire',inplace=True)



# There are few infinite values in the weather data,it will cause the training loss become NAN.Since the amount of np.inf is very few,it's simply replace by 0.

train.replace(np.inf,0,inplace=True)



# Transform percentage data to float

def get_percent(x):

    x = str(x)

    x = x.strip('%')

    x = int(x)/100

    return x



train.UrbanPopRate = train.UrbanPopRate.apply(lambda x:get_percent(x))



# Transform date type

def get_dt(x):

    return datetime.strptime(x,'%Y-%m-%d')



train.Date = train.Date.apply(lambda x:get_dt(x))
train.head()
for index,row in train.iterrows():

    if train.iloc[index].Province == train.iloc[index - 1].Province and train.iloc[index].ConfirmedCases < train.iloc[index-1].ConfirmedCases:

        train.iloc[index,4] = train.iloc[index-1,4]

    if train.iloc[index].Province == train.iloc[index - 1].Province and train.iloc[index].Fatalities < train.iloc[index-1].Fatalities:

        train.iloc[index,5] = train.iloc[index-1,5]
train_exam = train[['Country','Province','Date','ConfirmedCases','Fatalities']]

diff_df = pd.DataFrame(columns = ['Country','Province','Date','ConfirmedCases','Fatalities'])

for country in train_exam.Country.unique():

    for province in train_exam[train_exam.Country == country].Province.unique():

        province_df = train_exam.query(f"Country == '{country}' and Province == '{province}'")

        conf = province_df.ConfirmedCases

        fata = province_df.Fatalities

        diff_conf = conf.diff()

        diff_fata = fata.diff()

        province_df.ConfirmedCases = diff_conf

        province_df.Fatalities = diff_fata

        diff_df = pd.concat([diff_df,province_df],0)
print(sum(diff_df.ConfirmedCases < 0),sum(diff_df.Fatalities<0))
pd.set_option('mode.chained_assignment', None)
scale_train = pd.DataFrame(columns = ['Id_x', 'Province', 'Country', 'Date', 'ConfirmedCases', 'Fatalities',

       'Days_After_1stJan', 'Dayofweek', 'Month', 'Day', 'Population',

       'Density', 'Land_Area', 'Migrants', 'MedAge', 'UrbanPopRate', 'Id_y',

       'Lat', 'Long', 'temp', 'min', 'max', 'stp', 'slp', 'dewp', 'rh', 'ah',

       'wdsp', 'prcp', 'fog', 'API_beds'])

for country in train.Country.unique():

    for province in train.query(f"Country=='{country}'").Province.unique():

        province_df = train.query(f"Country=='{country}' and Province=='{province}'")

        province_confirm = province_df.ConfirmedCases

        province_fatalities = province_df.Fatalities

        province_confirm = np.array(province_confirm).reshape(-1,1)

        province_fatalities = np.array(province_confirm).reshape(-1,1)

        scaler1= preprocessing.MinMaxScaler()

        scaled_confirm = scaler1.fit_transform(province_confirm)

        scaler2 = preprocessing.MinMaxScaler()

        scaled_fata = scaler2.fit_transform(province_fatalities)

        province_df['ConfirmedCases'] = scaled_confirm

        province_df['Fatalities'] = scaled_fata

        scale_train = pd.concat((scale_train,province_df),axis = 0,sort=True)
trend_df = pd.DataFrame(columns={"infection_trend","fatality_trend","quarantine_trend","school_trend","total_population","expected_cases","expected_fatalities"})



train_df = scale_train

days_in_sequence = 14



trend_list = []



with tqdm(total=len(list(train_df.Country.unique()))) as pbar:

    for country in train_df.Country.unique():

        for province in train_df.query(f"Country=='{country}'").Province.unique():

            province_df = train_df.query(f"Country=='{country}' and Province=='{province}'")

            



            for i in range(0,len(province_df)):

                if i+days_in_sequence<=len(province_df):

                    #prepare all the trend inputs

                    infection_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].ConfirmedCases.values]

                    fatality_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].Fatalities.values]



                    #preparing all the stable inputs

                    days_after_1stJan = float(province_df.iloc[i].Days_After_1stJan)

                    dayofweek = float(province_df.iloc[i].Dayofweek)

                    month = float(province_df.iloc[i].Month)

                    day= float(province_df.iloc[i].Day)

                    population = float(province_df.iloc[i].Population)

                    density = float(province_df.iloc[i].Density)

                    land_area = float(province_df.iloc[i].Land_Area)

                    migrants = float(province_df.iloc[i].Migrants)

                    medage = float(province_df.iloc[i].MedAge)

                    urbanpoprate = float(province_df.iloc[i].UrbanPopRate)

                    beds = float(province_df.iloc[i].API_beds)



                    #True cases in i+days_in_sequence-1 day

                    expected_cases = float(province_df.iloc[i+days_in_sequence-1].ConfirmedCases)

                    expected_fatalities = float(province_df.iloc[i+days_in_sequence-1].Fatalities)



                    trend_list.append({"infection_trend":infection_trend,

                                     "fatality_trend":fatality_trend,

                                     "stable_inputs":[population,density,land_area,migrants,medage,urbanpoprate,beds],

                                     "expected_cases":expected_cases,

                                     "expected_fatalities":expected_fatalities})

        pbar.update(1)

trend_df = pd.DataFrame(trend_list)
trend_df["temporal_inputs"] = [np.asarray([trends["infection_trend"],trends["fatality_trend"]]) for idx,trends in trend_df.iterrows()]

trend_df = shuffle(trend_df)
trend_df.head()
# Only keeping 25 sequences where the number of cases stays at 0, as there were way too many of these samples in our dataset.

i=0

temp_df = pd.DataFrame()

for idx,row in trend_df.iterrows():

    if sum(row.infection_trend)>0:

        temp_df = temp_df.append(row)

    else:

        if i<25:

            temp_df = temp_df.append(row)

            i+=1

trend_df = temp_df
sequence_length = 13

training_percentage = 0.9

# The purpose of '-2'and'+2' is to make the number of samples in the training test set divisible by batchsize

training_item_count = int(len(trend_df)*training_percentage)

validation_item_count = len(trend_df)-int(len(trend_df)*training_percentage)

training_df = trend_df[:training_item_count-2]

validation_df = trend_df[training_item_count+2:]
X_temporal_train = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in training_df["temporal_inputs"].values]),(training_item_count-2,2,sequence_length)),(0,2,1) )).astype(np.float32)

X_stable_train = np.asarray([np.asarray(x) for x in training_df["stable_inputs"]]).astype(np.float32)

Y_cases_train = np.asarray([np.asarray(x) for x in training_df["expected_cases"]]).astype(np.float32)

Y_fatalities_train = np.asarray([np.asarray(x) for x in training_df["expected_fatalities"]]).astype(np.float32)



X_temporal_test = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in validation_df["temporal_inputs"]]),(validation_item_count-2,2,sequence_length)),(0,2,1)) ).astype(np.float32)

X_stable_test = np.asarray([np.asarray(x) for x in validation_df["stable_inputs"]]).astype(np.float32)

Y_cases_test = np.asarray([np.asarray(x) for x in validation_df["expected_cases"]]).astype(np.float32)

Y_fatalities_test = np.asarray([np.asarray(x) for x in validation_df["expected_fatalities"]]).astype(np.float32)



# Transform to tensor type

X_temporal_train = torch.from_numpy(X_temporal_train)

X_stable_train = torch.from_numpy(X_stable_train)

Y_cases_train = torch.from_numpy(Y_cases_train)

Y_fatalities_train = torch.from_numpy(Y_fatalities_train)



X_temporal_test = torch.from_numpy(X_temporal_test)

X_stable_test = torch.from_numpy(X_stable_test)

Y_cases_test = torch.from_numpy(Y_cases_test)

Y_fatalities_test = torch.from_numpy(Y_fatalities_test)



# Merge two objective values

Y_train = torch.cat((Y_cases_train.reshape(14770,1),Y_fatalities_train.reshape(14770,1)),1)

Y_test = torch.cat((Y_cases_test.reshape(1640,1),Y_fatalities_test.reshape(1640,1)),1)
print(len(X_temporal_train),len(X_temporal_test))
# Create train,test loader for training

class MyDataset(Dataset):

    def __init__(self, data1,data2, labels):

        self.trend= data1

        self.stable= data2

        self.labels = labels  



    def __getitem__(self, index):    

        trend,stable, labels = self.trend[index], self.stable[index], self.labels[index]

        return trend,stable,labels



    def __len__(self):

        return len(self.trend) 

    

train_ds = MyDataset(data1 = X_temporal_train,data2 = X_stable_train,labels = Y_train)

test_ds =MyDataset(data1 = X_temporal_test,data2 = X_stable_test,labels = Y_test)

train_loader = torch.utils.data.DataLoader(train_ds,batch_size = 10,shuffle=False)

test_loader = torch.utils.data.DataLoader(test_ds,batch_size = 10,shuffle=False)
# Define Model

class Net(nn.Module):

    def __init__(self):

            super(Net, self).__init__()

            self.lstm = nn.LSTM(2,32,1,dropout = 0.5)

            

            self.stable_full = nn.Linear(7,16)

            nn.init.kaiming_normal_(self.stable_full.weight)

            self.BN1 = nn.BatchNorm1d(16)

            self.stable_dropout = nn.Dropout(0.5)

            

            self.merge_full = nn.Linear(16+13*32,64)# stable:（5*16）  lstm:（13，5，32）

            nn.init.kaiming_normal_(self.merge_full.weight)

            self.BN2 = nn.BatchNorm1d(64)

            self.merge_dropout = nn.Dropout(0.3)

            self.merge_full2 = nn.Linear(64,2)

            nn.init.kaiming_normal_(self.merge_full2.weight)



    def reset_hidden(self):

        self.hidden = (torch.zeros(self.hidden[0].shape), torch.zeros(self.hidden[1].shape))

        

    def forward(self, x_trend,x_stable):

        batch_size = x_trend.reshape(13,-1,2).size(1)

        x_trend = x_trend.reshape(13,batch_size,2)

        x_trend, self.hidden = self.lstm(x_trend)

        

        x_stable = self.stable_dropout(F.relu(self.BN1(self.stable_full(x_stable))))

        

        s, b, h = x_trend.shape  #(seq, batch, hidden)

        x_trend = x_trend.view(b, s*h)

        x_merge = torch.cat((x_trend,x_stable),axis = 1)

        x_merge = F.relu(self.merge_full2(self.merge_dropout(F.relu(self.BN2(self.merge_full(x_merge))))))

        return x_merge
# Training Settings

model = Net()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

criterion = nn.MSELoss()
# Training process

def train_model(epoch):

    model.train()

    for batch_idx, (trend, stable, target) in enumerate(train_loader):

        trend, stable, target = Variable(trend), Variable(stable),Variable(target)

        optimizer.zero_grad()

        output = model(trend,stable)

        loss = criterion(output, target)  

        loss.backward()

        optimizer.step()

        if batch_idx % 300 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(trend), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.data))



def test_model(epoch):

    model.eval()

    test_loss = 0

    y_pred = []

    y_true = []

    for trend, stable, target in test_loader:

        trend,stable, target = Variable(trend),Variable(stable),Variable(target)

        output = model(trend,stable)

        test_loss += criterion(output, target).data

        y_pred.append(output)

        y_true.append(target)



    y_pred = torch.cat(y_pred, dim=0)

    y_true = torch.cat(y_true, dim=0)

    test_loss = test_loss

    test_loss /= len(test_loader) # loss function already averages over batch size

    MSE = mean_squared_error(y_true.detach().numpy(), y_pred.detach().numpy())

    print('\nTest set: Average loss: {:.4f}, MSE: {} \n'.format(

        test_loss, MSE 

        ))
for epoch in range(1, 21):

    train_model(epoch)

    test_model(epoch)
# In order to use query function,transform datetime to string

def get_str_date(x):

    x = str(x)[0:10]

    return x



scale_train.Date = scale_train.Date.apply(lambda x: get_str_date(x))
del scale_train['Id']
# read test_df and a new train_df

test_df = pd.read_csv('/kaggle/input/covid-with-weather-and-population/test_processed.csv')

train_df2 =  pd.read_csv('/kaggle/input/covid19-with-population/update_train_processed.csv')
test_df = test_df.query("Date > '2020-04-25'")
# same preprocess as before

train_df2.Country.replace('Cote d\'Ivoire','Cote d Ivoire',inplace=True)

train_df2.Province.replace('Cote d\'Ivoire','Cote d Ivoire',inplace=True)

train_df2.replace(np.inf,0,inplace=True)

train_df2.UrbanPopRate = train_df2.UrbanPopRate.apply(lambda x:get_percent(x))



test_df.Country.replace('Cote d\'Ivoire','Cote d Ivoire',inplace=True)

test_df.Province.replace('Cote d\'Ivoire','Cote d Ivoire',inplace=True)

test_df.replace(np.inf,0,inplace=True)

test_df.UrbanPopRate = test_df.UrbanPopRate.apply(lambda x:get_percent(x))
# make train dataframe and test dataframe have same columns

test_df['ConfirmedCases'] = np.NAN

test_df['Fatalities'] = np.NAN

test_df['Id_x'] = 0

test_df['Id_y'] = 0

test_df = test_df[list(scale_train.columns)]

# merge scale_train and test

total_df = pd.concat((scale_train,test_df),axis = 0)
def get_conf_scaler(country,province):

    train_df2_province = train_df2.query(f"Country == '{country}' and Province =='{province}'")

    train_df2_province_conf = train_df2_province['ConfirmedCases']

    train_df2_province_fata = train_df2_province['Fatalities']

    province_conf_scaler = preprocessing.MinMaxScaler()

    province_fata_scaler = preprocessing.MinMaxScaler()

    province_conf_scaler.fit(np.array(train_df2_province_conf).reshape(-1,1))

    province_fata_scaler.fit(np.array(train_df2_province_fata).reshape(-1,1))

    return province_conf_scaler,province_fata_scaler
def get_initial_input(country,province,start,end):

    input_province = total_df.query(f"Country =='{country}' and Province == '{province}' and Date>='{start}' and Date<='{end}'")

    input_list_province = []

    #prepare all the trend inputs

    infection_trend = [float(x) for x in input_province[:-1].ConfirmedCases.values]

    fatality_trend = [float(x) for x in input_province[:-1].Fatalities.values]



    #preparing all the stable inputs

    ##date inputs

    days_after_1stJan = float(input_province.iloc[-1].Days_After_1stJan)

    dayofweek = float(input_province.iloc[-1].Dayofweek)

    month = float(input_province.iloc[-1].Month)

    day= float(input_province.iloc[-1].Day)

    ##population inputs

    'Population','Density', 'Land_Area', 'Migrants', 'MedAge', 'UrbanPopRate'

    population = float(input_province.iloc[-1].Population)

    density = float(input_province.iloc[-1].Density)

    land_area = float(input_province.iloc[-1].Land_Area)

    migrants = float(input_province.iloc[-1].Migrants)

    medage = float(input_province.iloc[-1].MedAge)

    urbanpoprate = float(input_province.iloc[-1].UrbanPopRate)

    beds = float(input_province.iloc[-1].API_beds)



    input_list_province.append({"infection_trend":infection_trend,

                     "fatality_trend":fatality_trend,

                     "stable_inputs":[population,density,land_area,migrants,medage,urbanpoprate,beds],})

    input_df_province = pd.DataFrame(input_list_province)

    input_df_province["temporal_inputs"] = [np.asarray([input_df_province["infection_trend"],input_df_province["fatality_trend"]])]

    

    province_temporal_train = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in input_df_province["temporal_inputs"].values]),(1,2,sequence_length)),(2,0,1) )).astype(np.float32)

    province_stable_train = np.asarray([np.asarray(x) for x in input_df_province["stable_inputs"]]).astype(np.float32)

    province_temporal_train = torch.from_numpy(province_temporal_train)

    province_stable_train = torch.from_numpy(province_stable_train)

    return province_temporal_train,province_stable_train
def get_pred(country,province,trend,stable):

    conf_scaler,fata_scaler = get_conf_scaler(country,province)

    output = model(trend,stable)

    conf_output = conf_scaler.inverse_transform(output[0][0].detach().numpy().reshape(-1,1))

    fata_output = fata_scaler.inverse_transform(output[0][1].detach().numpy().reshape(-1,1))

    original_output = [conf_output,fata_output]

    return output,original_output
def get_pred_for_province(country,province):

    start_date = datetime.strptime('2020-04-13','%Y-%m-%d')

    end_date = datetime.strptime('2020-04-26','%Y-%m-%d')

    pred = []

    trend_input,stable_input = get_initial_input(country,province,str(start_date)[0:10],str(end_date)[0:10])

    for i in range(0,19):

        start = str(start_date+timedelta(days = i))[0:10]

        end = str(end_date+timedelta(days = i))[0:10]

        output,original_output = get_pred(country,province,trend_input,stable_input)

        pred.append([end,original_output[0],original_output[1]])

        trend_input = trend_input[1:]

        output_tensor = torch.as_tensor(output)

        new = torch.as_tensor(output_tensor.reshape(1,1,2))

        trend_input = torch.cat((trend_input,new),0)

    pred_for_province = pd.DataFrame(pred,columns=['Date','confirmed_pred','fata_pred'])

    pred_for_province['Province'] = province

    pred_for_province['Country'] = country

    pred_for_province = pred_for_province[['Country','Province','Date','confirmed_pred','fata_pred']]

    return pred_for_province
pred_table = pd.DataFrame(columns = ['Country','Province','Date','confirmed_pred','fata_pred'])

for country in test_df.Country.unique():

    for province in test_df.query(f"Country == '{country}'")['Province'].unique():

        province_pred = get_pred_for_province(country,province)

        pred_table = pd.concat((pred_table,province_pred),0)
pred_table
original_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
def get_dt(x):

    x = datetime.strptime(x,'%Y-%m-%d')

    return x
original_train.Date = original_train.Date.apply(lambda x:get_dt(x))

original_train.rename(columns = {'Province_State':'Province','Country_Region':'Country'},inplace = True)

for i in range(len(original_train)):

    if original_train.Province[i] is np.NaN:

        original_train.Province[i] = original_train.Country[i]

        

for i in range(len(original_train)):

    if original_train.Date[i]<datetime.strptime('2020-04-02','%Y-%m-%d') or original_train.Date[i]>datetime.strptime('2020-04-25','%Y-%m-%d'):

        original_train.drop(i,inplace=True)

        

del original_train['Id']

#del original_train['Country']



original_train.Province.replace('Cote d\'Ivoire','Cote d Ivoire',inplace=True)

#del pred_table['Country']

original_train.Date = original_train.Date.apply(lambda x:get_str_date(x))
def get_number(x):

    return x[0][0]

pred_table.confirmed_pred = pred_table.confirmed_pred.apply(lambda x:get_number(x))

pred_table.fata_pred = pred_table.fata_pred.apply(lambda x:get_number(x))



pred_table.rename(columns = {'confirmed_pred':'ConfirmedCases','fata_pred':'Fatalities'},inplace = True)

pred_table.Country.replace('Cote d Ivoire','Cote d\'Ivoire',inplace = True)

pred_table.Province.replace('Cote d Ivoire','Cote d\'Ivoire',inplace = True)
original_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

for i in range(len(original_test)):

    if original_test.iloc[i]['Province_State'] is np.NaN:

        original_test.iloc[i,1] = original_test.iloc[i,2]

        

original_test.rename(columns = {'Country_Region':'Country','Province_State':'Province'},inplace = True)
final = pd.concat([pred_table,original_train],axis = 0,sort = True)

final_submit = pd.merge(original_test,final,on = ['Country','Province','Date'],how = 'left')

submission = final_submit[['ForecastId','ConfirmedCases','Fatalities']]
submission
submission.to_csv('/kaggle/working/submission.csv',index = False)