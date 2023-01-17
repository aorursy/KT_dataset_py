# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import chainer as ch

from chainer import datasets

import chainer.functions as F

import chainer.links as L

from chainer import training

from chainer.training import extensions



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mfile='../input/mushrooms.csv'

data_array=np.genfromtxt(mfile,delimiter=',',dtype=str,skip_header=True)

df=pd.read_csv('../input/mushrooms.csv')

df.head()

df.shape
df.describe()
df.describe().iloc[1][1:].sum()   # summing along the "unique" row . all columns except the first one which corresponds to the class of the sample.
l=[x for x in df.columns[1:]]   # make a list of all the column names , all except the class column

print(l)

data_df=pd.get_dummies(df[l],drop_first=False)

data_df.shape
data_df.describe()
from sklearn.preprocessing import LabelEncoder

class_le=LabelEncoder()

y=class_le.fit_transform(df['class'].values)

y=y.reshape(y.shape[0],1)

assert(y.size==data_df.shape[0])              # I do this because I intend to create tuples of (data_df,y), so both the structures need to be of equal length

print(y)
TRAIN_SIZE_PERC=0.7



tuple_dataset=datasets.TupleDataset(data_df.values.astype(np.float32),y)

train,test=datasets.split_dataset_random(tuple_dataset,first_size=int(TRAIN_SIZE_PERC*len(tuple_dataset)))  # 70% of the data is used for training

print(len(train))

print(len(test))
BATCH_SIZE=120

train_iter=ch.iterators.SerialIterator(train,BATCH_SIZE)

test_iter=ch.iterators.SerialIterator(test,BATCH_SIZE,repeat=False,shuffle=False)
import chainer

from chainer import initializers

import chainer.functions as F



class CustomLinearLayer(chainer.Link):

    

    def __init__(self,n_in,n_out):

        super(CustomLinearLayer,self).__init__()

        with self.init_scope():

            self.W=chainer.Parameter(

                                        initializers.HeNormal(),      # He-initialization

                                        (n_out,n_in)                # W matrix is (n_out X n_in)

                                        

                                    )

            self.b=chainer.Parameter(

                                        initializers.Zero(),        # initialized to zero

                                        (n_out,)                   # bias is of shape (n_out,)

                                    )

            

    #forward propogation implementation:

    def forward(self,x):

        return F.linear(x,self.W,self.b)

    

import chainer

import chainer.functions as F

import chainer.links as L



class CustomMultiLayerPerceptron(chainer.Chain):

    

    def __init__(self,n_in,n_hidden,n_out):

        super(CustomMultiLayerPerceptron,self).__init__()

        with self.init_scope():

            self.layer1 = CustomLinearLayer(n_in,n_hidden)                                     # input layer

            self.layer2 = CustomLinearLayer(n_hidden,n_hidden)                                 # hidden layer

            self.layer3 = CustomLinearLayer(n_hidden,n_hidden)                                 # hidden layer

            self.layer4 = CustomLinearLayer(n_hidden,n_hidden)                                 # hidden layer

            self.layer5 = CustomLinearLayer(n_hidden,n_hidden)                                 # hidden layer

            self.layer6 = CustomLinearLayer(n_hidden,n_out)                                    # output layer

        

        #forward propagation

    def forward(self,*args):

        x=args[0]

        h1=F.relu(self.layer1(x))        # implements the  CustomLinearLayer link's forward propogation on x. i.e. h1=relu(x.W_1+b_1)

        h2=F.relu(self.layer2(h1))       # h2= relu( h1.W_2 + b_2)

        h3=F.relu(self.layer3(h2))       # h3= relu( h2.W_3 + b_3)

        h4=F.relu(self.layer4(h3))       # h4= relu( h3.W_4 + b_4)

        h5=F.relu(self.layer5(h4))       # h5= relu( h4.W_5 + b_5)

        #h6=F.sigmoid(self.layer6(h5))    # h6= sigmoid( h5.W_6 + b_6)

        h6=self.layer6(h5)    # h6=  h5.W_6 + b_6

        #print(h6)

        return h6

    
from chainer.functions.evaluation import accuracy

from chainer.functions.loss import softmax_cross_entropy

from chainer import link

from chainer import reporter



class CustomClassifier(link.Chain):

    def __init__(self,

                    predictor,                                            #predictor network that this classifier wraps

                    lossfun=softmax_cross_entropy.softmax_cross_entropy,  #the lossfunction it uses        

                    accfun=accuracy.accuracy,                              #the performance metric used

                    label_key=-1                                          #the location of the label in the input minibatch. (defaulted to the rightmost column)

                ):

        super(CustomClassifier,self).__init__()

        self.lossfun = lossfun

        self.accfun  = accfun

        self.y       = None                                               # the prediction from the last minibatch  y_hat

        self.loss    = None                                               #loss value for the last minibatch

        self.accuracy= None                                               #accuracy for the last minibatch

        self.label_key=label_key                                         # the location of the label in the input minibatch

        with self.init_scope():                                          # creates an initialization scope. See documentation for details.

            self.predictor = predictor

    

    

    def forward(self,*args,**kwargs):

        """

            Computes loss value for an input /label pair

            Computes accuracy 

            

            Args:

                args  : Input minibatch  

                kwargs: Input minibatch

            

        """

        self.y = None

        self.loss = None

        self.accuracy = None

        

        t=args[self.label_key]                                              #ground truth for the minibatch

    

        self.y = self.predictor(*args)                                 #get the output from the predictor

        self.loss=self.lossfun(self.y,t)                               #calculate the loss for this minibatch

        reporter.report({'loss':self.loss},self)

        self.accuracy = self.accfun(self.y,t)                          #the performance metric

        reporter.report({'accuracy':self.accuracy},self)

        

        return self.loss
model=CustomClassifier(CustomMultiLayerPerceptron(n_in=data_df.shape[1],n_hidden=data_df.shape[1]*3,n_out=1),

                       lossfun=F.sigmoid_cross_entropy,

                       accfun=F.binary_accuracy

                      )
optimizer=ch.optimizers.SGD(lr=0.001).setup(model)
updater=training.StandardUpdater(iterator=train_iter,optimizer=optimizer,device=-1) # set up the updater using 

                                                                          #the iterator and the optimizer
PERIOD=50                           

trainer=training.Trainer(updater,(PERIOD,'epoch'),out='result')
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

trainer.extend(extensions.dump_graph('main/loss'))

trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))

trainer.extend(extensions.LogReport())



if extensions.PlotReport.available():

    trainer.extend(

        extensions.PlotReport(['main/loss', 'validation/main/loss'],

                              'epoch', file_name='loss.png'))

    trainer.extend(

        extensions.PlotReport(

            ['main/accuracy', 'validation/main/accuracy'],

            'epoch', file_name='accuracy.png'))



    

trainer.extend(extensions.PrintReport(

    ['epoch', 'main/loss', 'validation/main/loss',

     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

trainer.run()
from chainer import serializers

serializers.save_hdf5('mushroom_model_01.hdf5',model)
#Load the trained model

nmod=CustomClassifier(CustomMultiLayerPerceptron(n_in=data_df.shape[1],n_hidden=data_df.shape[1]*3,n_out=1),

                       lossfun=F.sigmoid_cross_entropy,

                       accfun=F.binary_accuracy

                      )

serializers.load_hdf5('mushroom_model_01.hdf5',nmod)

#Check on test data

xtest=[x[0] for x in test]  # extract the test features

ytest=[x[1] for x in test]  # get the test labels

probs=nmod.predictor(np.array(xtest)).data # see Variable class. predictor(x) does the forward prop returning a Variable object h6. The .data is a member of the class





preds=np.where(probs>0,1,0)             # predictions thresholded at 0

from sklearn.metrics import roc_curve,auc,roc_auc_score

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



false_pos_rates,true_pos_rates,thresholds=roc_curve(ytest,probs)                       # we need to plot the roc . the y-axis are all true positives and the x-axis are all false positives.

print(false_pos_rates)                                                                 # we use the probabilities here while plotting the roc and NOT the predictions.



roc_auc=auc(false_pos_rates,true_pos_rates)

print(roc_auc)



roc_auc_sc=roc_auc_score(ytest,probs)

print(roc_auc_sc)



#plotting the roc 

trace=go.Scatter(

    x=false_pos_rates,

    y=true_pos_rates,

    mode='lines',

    name='ROC',

    

)

randomGuess=go.Line(

    x=[0,1],

    y=[0,1],

    line=dict(dash='dash'),

    name='random guess'

)



layout=go.Layout(

    annotations=[

        dict(

        x=0.2,

        y=0.6,

        text='AUC: \n '+str(roc_auc_sc),

        showarrow=False

        )

    ]

)

data=[trace,randomGuess]

fig=go.Figure(data=data,layout=layout)

py.iplot(fig)