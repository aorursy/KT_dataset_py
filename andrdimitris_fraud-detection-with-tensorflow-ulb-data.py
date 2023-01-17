import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sbn

from scipy.stats import ks_2samp,wasserstein_distance,energy_distance

from sklearn.pipeline import Pipeline,FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

import random

from imblearn.over_sampling import RandomOverSampler,SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.combine import SMOTEENN



from sklearn.metrics import precision_recall_curve,auc

from sklearn.model_selection import cross_val_score,StratifiedKFold,train_test_split,GridSearchCV,ParameterGrid





# Show plot output in the notebook

%matplotlib inline



sbn.set_style("darkgrid")
credit_card_data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
credit_card_data.head(30)
descriptive_stats=credit_card_data.describe()

descriptive_stats
df_info=credit_card_data.info()

df_info
# check more formally if there is any missing data

# isnull() checks each entry, and the following any() check across rows and columns

credit_card_data.isnull().any().any()
# Create the dataframe for the target

labels=credit_card_data[['Class']]



# Create the dataframe for the features

features=credit_card_data.drop(['Class'],axis=1)
labels['Class'].value_counts()
number_of_frauds=labels['Class'].value_counts()[1]

number_of_nonfrauds=len(labels['Class'])-number_of_frauds



fig,ax=plt.subplots(1,1,figsize=(10,10))

sbn.countplot(labels['Class'],ax=ax)

ax.set_title('Number of examples in each class (logarithmic scale)',fontsize=16)

ax.set_yscale('log')

ax.text(0-0.4,number_of_nonfrauds+10000,'Number of non-frauds: '+str(number_of_nonfrauds),fontsize=14)

ax.text(1-0.4,number_of_frauds+150,'Number of frauds: '+str(number_of_frauds),fontsize=14)





plt.show()
fraud_ratio=number_of_frauds/(len(labels['Class']))

print('Fraudulent examples comprise the {0:.4f} % of the total examples'.format(100*fraud_ratio))
descriptive_stats[['Time']]
time_in_hours=(features[['Time']]/(60.0*60.0)).astype(int)

time_in_hours.describe()
# Take the value counts from the time_in_hours and convert to dataframe.

# The index of the new dataframe now corresponds to the hour

transactions_per_hour=time_in_hours['Time'].value_counts(sort=False).to_frame()



# Reset the index to the dataframe so that the previous index becomes column

transactions_per_hour.reset_index(inplace=True)



# Change the name of the columns for better comprehension

transactions_per_hour.columns=['Hour','Transactions']



transactions_per_hour.head(10)
transactions_info=transactions_per_hour[['Transactions']].describe()

transactions_info
fig,axes=plt.subplots(1,2,figsize=(10,5))



ax=axes[0]



ax.hist(features['Time']/(60*60),bins=range(48))

ax.set_xticks([0,8,16,24,32,40,47])

ax.set_xlim([-1,49])

ax.set_title('Number of transactions per hour',fontsize=16)

ax.set_xlabel('Time [hours]',fontsize=14)

ax.set_ylabel('Counts' ,fontsize=14)



ax=axes[1]

ax.hist(transactions_per_hour['Transactions'],bins=10)

ax.set_title('Counts of the hourly transaction number',fontsize=16)

ax.set_xlabel('Hourly transactions',fontsize=14)





plt.subplots_adjust(wspace = 0.5)

plt.show()
fig = plt.figure(figsize=(15, 10))

grid = plt.GridSpec(3, 3, wspace=0.4, hspace=0.3)



ax1=fig.add_subplot(grid[0, 0])

ax2=fig.add_subplot(grid[1, 0])

ax3=fig.add_subplot(grid[2, 0])

ax4=fig.add_subplot(grid[:,1:])





ax=ax1



ax.hist(features['Time']/(60*60),bins=range(48))

ax.set_xticks([0,8,16,24,32,40,47])

ax.set_xlim([-1,49])

ax.set_title('Number of transactions per hour',fontsize=14)

ax.set_ylabel('Counts' ,fontsize=14)



ax=ax2



ax.hist(features[labels['Class']==0]['Time']/(60*60),bins=range(48))

ax.set_xticks([0,8,16,24,32,40,47])

ax.set_xlim([-1,49])

ax.set_title('Number of non-fraudulent transactions per hour',fontsize=14)

ax.set_ylabel('Counts' ,fontsize=14)



ax=ax3

ax.hist(features[labels['Class']==1]['Time']/(60*60),bins=range(48))

ax.set_xticks([0,8,16,24,32,40,47])

ax.set_xlim([-1,49])

ax.set_title('Number of fraudulent transactions per hour',fontsize=14)

ax.set_ylabel('Counts' ,fontsize=14)

ax.set_xlabel('Time [hours]',fontsize=14)



ax=ax4

sbn.distplot(features[labels['Class']==0]['Time']/(60*60),

             kde=True,hist=True,

             ax=ax,label='No Fraud',

             color='green',

             norm_hist=True

             )

sbn.distplot(features[labels['Class']==1]['Time']/(60*60),

             kde=True,

             hist=True,

             norm_hist=True,

             ax=ax,label='Fraud',color='red')



# sbn.distplot(features['Time']/(60*60),kde=True,hist=False,ax=ax,label='Total')

ax.set_xlabel('Time [hours]',fontsize=14)

ax.set_title('Distribution of frauds and non-frauds vs hours',fontsize=15)

# Use log scale to emphasize big (relative) differences

ax.set_yscale('log')

plt.show()
# Get the frauds

time_of_frauds=credit_card_data[credit_card_data['Class']==1][['Time']]

# Calculate the difference with previous row and add it as an additional column

time_of_frauds['Time difference']=time_of_frauds['Time'].diff()

time_of_frauds.head()
fig,axes=plt.subplots(1,2,figsize=(10,5))



# Set the bin edges to correspond to every 10 seconds

bins=[10*i for i in range(int(len(time_of_frauds)/10))]



# Select time difference up to which we zoom in

seconds_to_zoom=100

# Set the corresponding bins every 10 seconds

bins_zoom=[10*i for i in range(int(seconds_to_zoom/10)+1)]



ax=axes[0]

sbn.distplot(time_of_frauds['Time difference'].dropna(),ax=ax,norm_hist=False,kde=False,bins=bins)

ax.set_xlabel('Time difference [seconds]',fontsize=13)

ax.set_ylabel('Counts of fraudulent transactions',fontsize=13)

ax.set_title('Time difference distribution between\n consequtive frauds',fontsize=16)

ax.set_xticks(bins[::5])



ax=axes[1]

sbn.distplot(time_of_frauds['Time difference'].dropna(),ax=ax,norm_hist=False,kde=False,bins=bins_zoom)

ax.set_xlabel('Time difference [seconds]',fontsize=13)

ax.set_ylabel('Counts of fraudulent transactions',fontsize=13)

ax.set_title('Time difference distribution between\n consequtive frauds (zoomed in)',fontsize=16)

ax.set_xlim([0,100])

ax.set_xticks(bins_zoom)





plt.subplots_adjust(wspace = 0.5)



plt.show()
# Get the non - frauds

time_of_nonfrauds=credit_card_data[credit_card_data['Class']==0][['Time']]

# Calculate the difference with previous row and add it as an additional column

time_of_nonfrauds['Time difference']=time_of_nonfrauds['Time'].diff()

time_of_nonfrauds.head()
fig,axes=plt.subplots(1,2,figsize=(10,5))



# Set the bin edges to correspond to every 10 seconds

bins=[10*i for i in range(int(len(time_of_frauds)/10))]



# Select time difference up to which we zoom in

seconds_to_zoom=100

# Set the corresponding bins every 10 seconds

bins_zoom=[10*i for i in range(int(seconds_to_zoom/10)+1)]



ax=axes[0]

sbn.distplot(time_of_nonfrauds['Time difference'].dropna(),ax=ax,norm_hist=False,kde=False,bins=bins)

ax.set_xlabel('Time difference [seconds]',fontsize=13)

ax.set_ylabel('Counts of non fraudulent transactions',fontsize=13)

ax.set_title('Time difference distribution between\n consequtive non frauds',fontsize=16)

ax.set_xticks(bins[::5])



ax=axes[1]

sbn.distplot(time_of_nonfrauds['Time difference'].dropna(),ax=ax,norm_hist=False,kde=False,bins=bins_zoom)

ax.set_xlabel('Time difference [seconds]',fontsize=13)

ax.set_ylabel('Counts of non fraudulent transactions',fontsize=13)

ax.set_title('Time difference distribution between\n consequtive non frauds (zoomed in)',fontsize=16)

ax.set_xlim([0,100])

ax.set_xticks(bins_zoom)





plt.subplots_adjust(wspace = 0.5)



plt.show()
# Get the non - frauds

time_diff=credit_card_data[['Time','Class']]

# Calculate the difference with previous row and add it as an additional column

time_diff['Time difference']=time_diff['Time'].diff()

time_diff[['Time difference']].describe()
fig,ax=plt.subplots(1,1)

sbn.distplot(time_diff[time_diff['Class']==0]['Time difference'].dropna(),

             kde=False,bins=[5*i for i in range(7)],

             label='Non-fraud')

sbn.distplot(time_diff[time_diff['Class']==1]['Time difference'].dropna(),

             kde=False,color='green',

             bins=[5*i for i in range(7)],

             label='Fraud')

ax.set_yscale('log')

ax.set_xlabel('Time difference [seconds]',fontsize=13)

ax.set_ylabel('Counts of transactions',fontsize=13)

ax.set_title('Type of transactions vs. time difference \nbetween consequtive transactions',fontsize=16)

ax.legend()

plt.show()
features[['Amount']].describe()
fig,axes=plt.subplots(1,2,figsize=(10,6))



ax=axes[0]



sbn.distplot(credit_card_data[credit_card_data['Class']==0]['Amount'],

             ax=ax,

             kde=False,

             norm_hist=False,

             bins=20,

             color='green',

             label='No Fraud')



sbn.distplot(credit_card_data[credit_card_data['Class']==1]['Amount'],

             ax=ax,

             kde=False,

             norm_hist=False,

             bins=20,

             color='red',

             label='Fraud')



ax.set_yscale('log')

ax.set_title('Histogram of "Amount" for each class',fontsize=16)

ax.set_xlabel('Amount',fontsize=13)

ax.set_ylabel('Counts',fontsize=13)



ax=axes[1]



sbn.distplot(credit_card_data[credit_card_data['Class']==0]['Amount'],

             ax=ax,

             kde=False,

             norm_hist=True,

             bins=20,

             color='green',

             label='No Fraud')



sbn.distplot(credit_card_data[credit_card_data['Class']==1]['Amount'],

             ax=ax,

             kde=False,

             norm_hist=True,

             bins=20,

             color='red',

             label='Fraud')



ax.set_yscale('log')

ax.set_title('Histogram of "Amount" for each class',fontsize=16)

ax.set_xlabel('Amount',fontsize=13)

ax.set_ylabel('Density',fontsize=13)





plt.subplots_adjust(wspace = 0.5)



plt.show()
print('No Fraud')

features[labels['Class']==0][['Amount']].describe()
print('Fraud')

features[labels['Class']==1][['Amount']].describe()
fig = plt.figure(figsize=(15, 10))

grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)





ax1=fig.add_subplot(grid[0, :])

ax2=fig.add_subplot(grid[1, 0])

ax3=fig.add_subplot(grid[1, 1])





sbn.scatterplot(x='Time',y='Amount',data=credit_card_data,ax=ax1,alpha=0.35)

ax1.set_title('Amount vs time for all transactions',fontsize=16)

ax1.set_xlabel('Time [seconds]',fontsize=13)

ax1.set_ylabel('Amount',fontsize=13)







sbn.scatterplot(x='Time',y='Amount',data=credit_card_data[credit_card_data['Class']==0],ax=ax2,color='green',alpha=0.35)

ax2.set_title('Amount vs time for normal transactions',fontsize=16)

ax2.set_xlabel('Time [seconds]',fontsize=13)

ax2.set_ylabel('Amount',fontsize=13)



sbn.scatterplot(x='Time',y='Amount',data=credit_card_data[credit_card_data['Class']==1],ax=ax3,color='red',alpha=0.35)

ax3.set_title('Amount vs time for fraudulent transactions',fontsize=16)

ax3.set_xlabel('Time [seconds]',fontsize=13)

ax3.set_ylabel('Amount',fontsize=13)



plt.show()

anonymised_features=features.drop(['Time','Amount'],axis=1)
anonymised_features.describe()
anonymised_features=features.drop(['Time','Amount'],axis=1)



plt.figure(figsize=(12,28*4))

grid = plt.GridSpec(28, 1)



ks_distances=[]

emd_distances=[]

ed_distances=[]



for idx,feature in enumerate(anonymised_features.columns):

    

    plt.subplot(grid[idx])

        

    sbn.distplot(anonymised_features[labels['Class']==0][feature],

                 kde=True,

                 color='green',

                 label='No Fraud',bins=30)

    

    sbn.distplot(anonymised_features[labels['Class']==1][feature],

                 kde=True,

                 color='red',

                 label='Fraud',bins=30)

    

    

    ks=ks_2samp(anonymised_features[labels['Class']==1][feature].values,

                anonymised_features[labels['Class']==0][feature].values)

        

    emd=wasserstein_distance(anonymised_features[labels['Class']==1][feature].values,

                             anonymised_features[labels['Class']==0][feature].values)

    

    ed=energy_distance(anonymised_features[labels['Class']==1][feature].values,

                       anonymised_features[labels['Class']==0][feature].values)

    

    ks_distances.append(ks[0])

    emd_distances.append(emd)

    ed_distances.append(ed)

        

    plt.title(feature+': KS: {0:.2f}, EMD: {1:.2f}, ED: {2:.2f}'.format(ks[0],emd,ed),fontsize=20)

    plt.xlabel(feature,fontsize=18)

    plt.legend()

    

plt.subplots_adjust(hspace = 0.5)



plt.show()
fig,axes=plt.subplots(3,1,figsize=(15,20),sharex=False)



ax=axes[0]



sbn.barplot(x=np.arange(28),y=ks_distances,ax=ax)

ax.set_title('Kolmogorov-Smirnov Statistic',fontsize=16)

ax.set_xticklabels(anonymised_features.columns.values)





ax=axes[1]



sbn.barplot(x=np.arange(28),y=emd_distances,ax=ax)

ax.set_title('Wasserstein distance',fontsize=16)

ax.set_xticklabels(anonymised_features.columns.values)



ax=axes[2]



sbn.barplot(x=np.arange(28),y=ks_distances,ax=ax)

ax.set_title('Energy distance',fontsize=16)



ax.set_xticklabels(anonymised_features.columns.values)





plt.show()
class ColumnSelector(BaseEstimator, TransformerMixin):

    

    """

    (Transformer)

    Class that implements selection of specific columns.

    The desired column or columns are passed as an argument to the constructor





    """



    def __init__(self, cols):



        """

        :param cols: desired columns to keep

        :return: the transformed dataframe



        """



        self.cols = cols



    def fit(self, X, y=None):



        """

        :param X: dataframe

        :param y: none

        :return: self



        """



        return self



    def transform(self, X):



        """



        :param X: dataframe

        :return: the dataframe with only the selected cols



        """



        # First check if X is a pandas DataFrame



        assert isinstance(X, pd.DataFrame)



        try:



            # Return the desired columns if all of them exist in the dataframe X

            return X[self.cols]



        except KeyError:



            # Find which are the missing columns, i.e. desired cols to keep that do not exist in the dataframe

            missing_cols = list(set(self.cols) - set(X.columns))



            raise KeyError("The columns: %s do not exist in the data" % missing_cols)

            

            

class Scaler(BaseEstimator, TransformerMixin):

    

    """

    (Transformer)

    Class that implements scaling.

    

    method: string, either 'normalize' or 'standardize'

        

    """



    def __init__(self, method):



        self.method = method



    def fit(self, X, y=None):



        return self



    def transform(self, X):



        if self.method == 'normalize':



            return (X - X.min()) / (X.max() - X.min())



        elif self.method == 'standardize':



            return (X - X.mean()) / X.std()

        
class DataPrep(object):

    

    """

    (Transformer)

    Prepare data and implement pipeline





    columns_to_keep: list of which columns of the dataframe to keep

    normalization_method:string, 'normalize' or 'standardize' denoting the scaling type

    

    """



    ### Pass the desired arguments to the constructor



    def __init__(self, columns_to_keep,normalization_method):

        self.columns_to_keep = columns_to_keep

        self.normalization_method=normalization_method



    def pipeline_creator(self):

        

        """

        

        The pipelines are "trivial", but could be extended by adding more functionalities (e.g. select which cols to 

        normalize, deal with categorical cols...)

        

        """

        

        #Data Selection



        data_select_pipeline = Pipeline([



            ('Columns to keep', ColumnSelector(self.columns_to_keep)),



        ])



    

        norm_pipeline = Pipeline([

            

            ('Custom Scaling', Scaler(self.normalization_method))

        ])

        

        

        preprocess_pipeline=Pipeline([

            

            ('Data selection',data_select_pipeline),

            ('Data scaling',norm_pipeline)



        ])

        

        

        return preprocess_pipeline
class MyEstimator(BaseEstimator):

    

    """

    (Estimator)

    

    Class that implements our custom estimator

    

    """



    def __init__(self,sampling_method,learning_rate,batch_size,num_epochs,keep_prob,model_dir,model_name):

        

        """

        sampling_method:string, 'over' ,'under', 'over_SMOTE' ,'both'

        learning_rate: float, the learning rate of the algorithm

        batch_size:the batch size to feed each step of optimization

        num_epochs: int, iteration over the dataset

        keep_prob: probability to keep nodes at the dropout layer (if any)

        model_dir: where to save output for tensorboard visualization

        

        

        """

        

        self.sampling_method = sampling_method        

        self.learning_rate = learning_rate

        self.batch_size=batch_size

        self.num_epochs=num_epochs

        self.model_dir=model_dir

        self.keep_prob=keep_prob

        self.model_name=model_name

    

#     def make_tf_dataset(self,X,y,num_epochs,batch_size):

        

#         dataset=tf.data.Dataset.from_tensor_slices({'features':X,'labels':y})

#         dataset.shuffle(X.shape[1])

#         dataset.batch(batch_size)

#         dataset.repeat(num_epochs)

        

#         return dataset

    

    def resample_dataset(self,X,y):

        

        """

        Method that implements the resampling. X and y correspond to data after train/test split

        sampling_strategy=1 denotes that the ratio of classes will be equal in the resampled datasets

        

        Beware, in oversampling, huge datasets could result

        

        """

        

        if self.sampling_method=='over':

            

            sampler=RandomOverSampler(sampling_strategy=0.03,random_state=0)

            X_resampled,y_resampled=sampler.fit_resample(X,y)

            

        elif self.sampling_method=='under':

            

            sampler=RandomUnderSampler(sampling_strategy=0.03,random_state=0)

            X_resampled,y_resampled=sampler.fit_resample(X,y)

            

        elif self.sampling_method=='over_SMOTE':

            

            sampler=SMOTE(sampling_strategy=1,random_state=0)

            X_resampled, y_resampled = sampler.fit_resample(X, y)

            

        elif self.sampling_method=='both':

            

            sampler=SMOTEENN(sampling_strategy=0.03,random_state=0)

            X_resampled, y_resampled = sampler.fit_resample(X, y['Class'].values)

        

        else:

            

            print('No resampling is used!')

            

            X_resampled=X

            y_resampled=y

        

        

        return X_resampled, y_resampled

        

        

    

        

    def fit(self,X,y):

        

        """

        

        The fit method should implement training.

        We resample the data, create the graph, run the training for num_epochs and for each step we feed in

        batch_size of examples. We finally save the model for usage by predict and score methods

        

        

        """

        

        #for debugging

        print('\n Number of input examples for fit: '+str(X.shape[0]))

        

        # undersampling/oversampling

        

        X_resampled, y_resampled=self.resample_dataset(X,y)

        

        y_resampled=np.asarray(y_resampled)

        y_resampled=np.reshape(y_resampled,(-1,))

        

        # Had to use float16 in order not to run out of memory; maybe it is not an issue if you run python script not in Jupyter

        X_resampled=X_resampled.astype(np.float16)

        y_resampled=y_resampled.astype(np.int16)

        

#         print(X_resampled.shape)

#         print(X_resampled.dtype)



#         print(y_resampled.shape)

        

        print('\n Number of examples after resampling: '+str(X_resampled.shape[0]))

        

        

        # Create the graph (placeholders, variables, ops...) and then train the model for num_epochs

        

        # Reset graph so no overlapping names occur accross multiple runs

        tf.reset_default_graph()



        """

        

        Create placeholders for the input. Every row of features correspond to the same row at output

        The dimensions can be transposed, and this will affect how the matrix multiplication is done, so a good practice

        would be to keep track of the dimensions, which we will append as comments in the code

        

        As a note, the values of the placeholders are not saved with the tf.train.Saver().save() method

        

        """

        # Features placeholder will have shape (batch_size,number_of_features)

        

        features=tf.placeholder(tf.float16,[None,X_resampled.shape[1]],name='features')

        

        # Labels placeholder will have shape(number_of_features,1)

        

        labels=tf.placeholder(tf.int32,[None,],name='labels')

        

        # We create a placeholder for the keep probability of the dropout layer.

        # !!! SET TO 1 DURING DEV/TEST

        

        prob_to_keep=tf.placeholder(tf.float16,name='keep_prob')

        

        '''

        As an example algorithm, we use a neural net (NN) with multiple layers. 

        Logistic regression can be modeled with a NN of 1 (hidden) layer with 1 node

        

        By using more hidden layers, we let the network learn more complex functions of the input 

        than just a linear combination of it

        

        '''

        

        # Define the number of hidden layers and nodes. 

        # As a rule, we keep the ratio of nodes between consequtive hidden layers fixed

        

        ratio=1.5

        

        # The input nodes regard the input layer and correspond to the number of features

        

        input_nodes=X_resampled.shape[1]

        

        # Nodes in the first hidden layer

        

        hidden_nodes_1=8

        

        # Nodes in the second hidden layer

        

        hidden_nodes_2=round(hidden_nodes_1*ratio)

        

        # Nodes in the third hidden layer

        

        hidden_nodes_3=round(hidden_nodes_2*ratio)

        

        # Output nodes: we have class 0 and class 1

        # With one output node, the output of the network will yield the probability of class=1, i.e. the probability to have fraud.

        # Since the classes are mutually exclusive, the probability of no fraud will be 1-P(fraud)

        

        output_nodes=2

        

        # Construct hidden layer 1

        

        with tf.name_scope('Hidden_Layer_1'):

        

            W1 = tf.get_variable('W1',[input_nodes,hidden_nodes_1],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)

            b1 = tf.get_variable('B1',[hidden_nodes_1],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)

            

            # out_1 will be matrix multiplication (batch_size,number_of_features)*(input_nodes,hidden_nodes_1)

            # out_1 will be of shape (batch_size,hidden_nodes_1)

            

            # We use ReLU non-linearities to speed-up learning

            out_1=tf.nn.relu(tf.matmul(features,W1)+b1,name='out_1')

            

            

            tf.summary.histogram('Weights',W1)

            tf.summary.histogram('Biases',b1)

            tf.summary.histogram('Activations',out_1)

            

        

        # Construct hidden layer 2

         

        with tf.name_scope('Hidden_Layer_2'):

            

            W2=tf.get_variable('W2',[hidden_nodes_1,hidden_nodes_2],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)

            

            b2 = tf.get_variable('B2',[hidden_nodes_2],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)



            

            # out_2 will be matrix multiplication (batch_size,hidden_nodes_1)*(hidden_nodes_1,hidden_nodes_2)

            # out_2 will be of shape (batch_size,hidden_nodes_2)



            out_2=tf.nn.relu(tf.matmul(out_1,W2)+b2,name='out_2')

            

            out_2=tf.nn.dropout(out_2,prob_to_keep,name='out_2_dropout')

        

            tf.summary.histogram('Weights',W2)

            tf.summary.histogram('Biases',b2)

            tf.summary.histogram('Activations',out_2)

            

        # Construct hidden layer 3 (comment out if needed)

        '''

        with tf.name_scope('Hidden_Layer_3'):

            

            W3=tf.get_variable('W3',[hidden_nodes_2,hidden_nodes_3],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)

            

            b3 = tf.get_variable('B3',[hidden_nodes_3],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)



            # out_3 will be matrix multiplication (batch_size,hidden_nodes_2)*(hidden_nodes_2,hidden_nodes_3)

            # out_3 will be of shape (batch_size,hidden_nodes_3)

            

            out_3=tf.nn.relu(tf.matmul(out_2,W3)+b3,name='out_3')

            

            out_3=tf.nn.dropout(out_3,prob_to_keep,name='out_3_dropout')

            

            

            tf.summary.histogram('Weights',W3)

            tf.summary.histogram('Biases',b3)

            tf.summary.histogram('Activations',out_3)

        '''    

        

        # construct hidden layer 4 (modify the dimensions accordingly)

        

        with tf.name_scope('Output_Layer'):

            

            W4 = tf.get_variable('W4',[hidden_nodes_2,output_nodes],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)

            b4 = tf.get_variable('B4',[output_nodes],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)



            

            # out_4 will be matrix multiplication (batch_size,hidden_nodes_3)*(hidden_nodes_3,2)

            # out_4 will be of shape (batch_size,2)

            

            # We do not apply any non-linearity to the output, as this will be taken care by the loss operation



            out_4=tf.add(tf.matmul(out_2,W4),b4,name='out_4')

            

            

            tf.summary.histogram('Weights',W4)

            tf.summary.histogram('Biases',b4)

            tf.summary.histogram('Activations',out_4)

               

            

        with tf.name_scope('Loss'):

            

            loss=tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=out_4),name='loss')

            

            # We decide to calculate and keep only the loss. 

            # Accuracy is not so good a metric when we have imbalanced datasets

            

            tf.summary.scalar('loss',loss)

        

       

        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate,epsilon=0.05,name='optimizer')

        train_op=optimizer.minimize(loss,name='training_objective')

                      

        # Define an operation to initialize global variables    

        init=tf.global_variables_initializer()

        

        merged_summary_op=tf.summary.merge_all()

        

        with tf.Session() as sess:

            

            # Initialize the global variables

            

            sess.run(init)

            

            print('\n Training has started...')

            

            

            # Define operation to write to tensorboard

            

            summary_writer = tf.summary.FileWriter(self.model_dir, graph=tf.get_default_graph())

            

            

            # Train the network for num_epochs (iterations over dataset)

            

            for epoch in range(self.num_epochs):

                

                # For each epoch, reset the training cost

                

                batch_cost=0

                

                num_batches=int(X_resampled.shape[0]/self.batch_size)

                

                # Each optimization step will be done using a batch of the data of size batch_size

                

                for batch in range(num_batches):                    

                    

                    batch_x=X_resampled[batch*self.batch_size:(batch+1)*self.batch_size,:]

                    batch_y=y_resampled[batch*self.batch_size:(batch+1)*self.batch_size]

                    

                                    

                    _,temp_cost,summary=sess.run([train_op,loss,merged_summary_op],

                                                 feed_dict={features:batch_x,labels:batch_y,prob_to_keep:self.keep_prob})

                    

                    # Calculate an average cost over the number of batches

                    

                    batch_cost+=temp_cost/num_batches

                                        

                    # Write all the selected variables for every iteration 

                    summary_writer.add_summary(summary=summary, global_step=epoch * num_batches + batch )

                

                    to_string='Minibatch:'+str(batch)+'/'+str(num_batches)

                

                    sys.stdout.write('\r'+to_string)

                

                # Print cost (training loss) at regular intervals

                if epoch % 10 ==0:

                    

                    print('\nTraining cost at epoch {} : {}'.format(epoch,batch_cost))

                    

            

            print('\n Optimization finished! \n')

        

        

        

        # Save the model. We have to do that inside the session.

        # We decide to keep the last iteration

        

            saver=tf.train.Saver()

            print('\n Saving trained model...\n')

            saver.save(sess,'./'+self.model_name)

        

        

    def predict(self,X):

        

        """

        Predict the output probabilities from a trained model. First we restore the model and load any desired tensors

        and nodes. We add on top the softmax layer, create the corresponding op and run the session

        

        """

        

        with tf.Session() as sess:

            

            

        # this loads the graph

            saver = tf.train.import_meta_graph(self.model_name+'.meta')

        

        # this gets all the saved variables

            saver.restore(sess,tf.train.latest_checkpoint('./'))

            

        # get the graph

        

            graph=tf.get_default_graph()

            

            # get placeholder tensors to feed new values

            features=graph.get_tensor_by_name('features:0')

#             labels=graph.get_tensor_by_name('labels:0')

            keep_prob=graph.get_tensor_by_name('keep_prob:0')

            

            # get the desired operation to restore. this will be the output of the last layer

            op_to_restore=graph.get_tensor_by_name('Output_Layer/out_4:0')

            

            # For prediction the keep_prob of the dropout layer will be equal to 1, i.e. no dropout

            logits=sess.run(op_to_restore,feed_dict={features:X,keep_prob:1.0})

            

            # The output of this operation needs to be passed to a softmax layer in order to get as output probabilities

            # Define the necessary op

            softmax=tf.nn.softmax(logits)

            

            # Run the op

            probabilities=sess.run(softmax)

            

            # The output will be of shape (num_examples,2)

            # The first column corresponds to P(Class=0) , that is no fraud

            # The second column corresponds to P(Class=1), that is fraud

            

            return probabilities

            

    

    

    

    '''



    Our custom estimator needs to implement a score method that will be used 

    to select the best custom score using grid search

    

    For such imbalanced dataset that we have, a good metric, 

    as alredy discussed, will be the area under the precision-recall curve

        

    '''

    

    def score(self,X,y):

        

        """

        

        Scorer function to be used for selection of the best parameters.

        Our scorer function will calculate the area under the precison recall curve

        

        

        """

        

        print('\n Scoring using '+str(X.shape[0])+' examples')

        

        # Get the probabilities for each class

        probs=self.predict(X)

        

        # Get the probabilities for fraud for each example

        fraud_probs=probs[:,1]

        

        print('\n Got probabilities for '+str(X.shape[0])+' examples')

        

        # Define the operation to get the area under the precision-recall curve

        

        with tf.name_scope('Scoring'):

        

            area=tf.metrics.auc(labels=y,

                            predictions=fraud_probs,

                            curve='PR',

                            summation_method='careful_interpolation',

                            name='AUCPRC')

        

        # Define an initialization operation on the global AND local variables

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        

        # Create session and run

        

        with tf.Session() as sess:

            

            # Initialize the variables

            sess.run(init)

            

            # Run the area node to calculate AUCPRC

            out=sess.run(area)

        

        return out[1]

    

                
# use 'under' for resampling to get results quicker, as here the dataset is smaller



params={'sampling_method':['under'],'learning_rate':[0.003,0.01],'batch_size':[128],

        'num_epochs':[100],'keep_prob':[0.35,0.7],'cols':['full','discard'],'cv_splits':[3]}



params=ParameterGrid(params)

scores_runs=[]



for run,item in enumerate(params):

    

    if item['cols']=='full':

        

        cols=features.columns

        pipeline_data=DataPrep(cols,'standardize').pipeline_creator()

        

    elif item['cols']=='discard':

        # drop the anonymised features with the smallest wasserstein distance.

        cols=features.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V21','V19','V15','V13','V6'],axis=1).columns

        pipeline_data=DataPrep(cols,'standardize').pipeline_creator()



    

    folds=StratifiedKFold(n_splits=item['cv_splits'],shuffle=True,random_state=42)

        

    sampling_method=item['sampling_method']

    learning_rate=item['learning_rate']

    batch_size=item['batch_size']

    num_epochs=item['num_epochs']

    keep_prob=item['keep_prob']

    

    full_pipeline=Pipeline([

    

    ('data prep',pipeline_data),

    ('estimator',MyEstimator(sampling_method,learning_rate,batch_size,num_epochs,keep_prob,model_dir='./tmp_over'+str(run),

    model_name='over-model-run'+str(run)))



    ])

        

    scores=cross_val_score(full_pipeline,features,labels,cv=folds.split(features,labels))

    

    scores_runs.append(scores)

    

    np.save('scores'+str(run),scores)

    
# The parameters used when undersampling

params_under={'sampling_method':['under'],'learning_rate':[0.003,0.01],'batch_size':[128],

        'num_epochs':[100],'keep_prob':[0.35,0.7],'cols':['full','discard'],'cv_splits':[3]}





params=ParameterGrid(params_under)



# make a dataframe containing as entries the parameters dictionary for each run

results=pd.DataFrame([item for item in params])
scores=[]



for run,item in enumerate(params):

    run_scores=np.load('scores'+str(run)+'.npy')

    scores.append(run_scores)



    

scores_array=np.array(scores)

scores_array
results['cv_1']=scores_array[:,0]

results['cv_2']=scores_array[:,1]

results['cv_3']=scores_array[:,2]
results['mean_score']=results[['cv_1','cv_2','cv_3']].mean(axis=1)

results