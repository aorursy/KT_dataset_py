import pandas as pd
pd.options.display.float_format = "{:,.4f}".format
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For data scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler

#For classes re labeling
from sklearn.preprocessing import LabelBinarizer

# For Dimension Reduction
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap
from umap import UMAP

## For building auto encoder
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# For Imbalanced classes
from imblearn.over_sampling import SMOTE

# For Machine Learning modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# For Cross vaidation
from sklearn.model_selection import cross_validate
plt.style.use('fivethirtyeight')
import random

# silence NumbaPerformanceWarning
import warnings
from numba.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

train = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')
test = pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')
lb = LabelBinarizer()
train['LABEL'] = lb.fit_transform(train['LABEL'])
test['LABEL'] = lb.transform(test['LABEL'])
print(train.shape)
print(test.shape)
train.dtypes
train.describe().T
train.isna().sum()
train['LABEL'].value_counts().plot(kind = 'bar', title = 'Class Distributions \n (0: Not Exoplanet || 1: Exoplanet)', rot=0)
# PROBLEM OF IMBALANCED CLASS
# RUNNING ANY ALGORITHM WILL RESULT IN A ACCURACY MEASURE THAT IS HIGHLY INFLUENCED BY DOMINATING CLASS
#Visualizing the the 5 five rercords
plt.figure(figsize=(25,10))
plt.title('Distribution of flux values', fontsize=15)
plt.xlabel('Flux values')
plt.ylabel('Flux intensity')
plt.plot(train.iloc[0,])
plt.plot(train.iloc[1,])
plt.plot(train.iloc[2,])
plt.plot(train.iloc[3,])
plt.plot(train.iloc[4,])
plt.legend(('Data1', 'Data2', 'Data3', 'Data4', 'Data5'))
plt.show()
Exoplanet = train[train['LABEL']==1]
Not_Exoplanet = train[train['LABEL']==0]
#Visualizing the Exoplanets data, Since columns are 3198, bining is one way to see the gaussian hist
for i in range(5):
    flux = random.choice(Exoplanet.index)
    plt.figure(figsize=(20,10))
    plt.hist(Exoplanet.iloc[flux,:], bins=100)
    plt.title("Gaussian Histogram of Exoplanets")
    plt.xlabel("Flux values")
    plt.show()
#Visualizing the Non Exoplanets data, Since columns are 3198, bining is one way to see the gaussian hist
for i in range(5):
    flux = random.choice(Not_Exoplanet.index)
    plt.figure(figsize=(20,10))
    plt.hist(Not_Exoplanet.iloc[flux,:], bins=100)
    plt.title("Gaussian Histogram of Non Exoplanets")
    plt.xlabel("Flux values")
    plt.show()
#pd.to_numeric(train.iloc[:,0], downcast='integer')
#train.iloc[:,1:].apply(pd.to_numeric(train.iloc[:,x:], downcast='float'))
def memory_manager(df):
    for col, types in df.dtypes.iteritems():
        if types == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')    
        elif types == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float') 
    return df
# Normalizing the Dataset
#For this just for exploring best technique we will use all the normalization techniques
def data_normalizing(df_train,df_test, option):
    if option ==  1:
        scale = StandardScaler()  
        df_train = scale.fit_transform(df_train)
        df_test = scale.transform(df_test)
    elif option == 2:
        scale = MinMaxScaler()
        df_train = scale.fit_transform(df_train)
        df_test = scale.transform(df_test)
    elif option == 3:
        scale = MaxAbsScaler()
        df_train = scale.fit_transform(df_train)
        df_test = scale.transform(df_test)
    elif option ==  4:
        scale = Normalizer()
        df_train = scale.fit_transform(df_train)
        df_test = scale.transform(df_test)
    elif option ==  5:
        scale = RobustScaler()
        df_train = scale.fit_transform(df_train)
        df_test = scale.transform(df_test)
    elif option == 6:
        scale = 'None'
        df_train
        df_test 
    else:
        print('Enter Valid option')
        
    
    return df_train, df_test , str(scale).replace("()","")
        
# High Dimensional issues
# No of columns is >3000
#Reduce dimension using technique which maximizes the result
def auto_encoder(df,df_test, component):
    # input placeholder
    input_data = Input(shape=df.shape[1:]) # 6 is the number of features/columns

    encoded = Dense(2048, activation = 'relu')(input_data)
    encoded = Dense(1024, activation = 'relu')(encoded)
    encoded = Dense(512, activation = 'relu')(encoded)
    encoded = Dense(256, activation = 'relu')(encoded)
    encoded = Dense(128, activation = 'relu')(encoded)
    encoded = Dense(64, activation = 'relu')(encoded)
    encoded = Dense(component, activation = 'relu')(encoded)

    decoded = Dense(64, activation = 'relu')(encoded)
    decoded = Dense(128, activation = 'relu')(decoded)
    decoded = Dense(256, activation = 'relu')(decoded)
    decoded = Dense(512, activation = 'relu')(decoded)
    decoded = Dense(1024, activation = 'relu')(decoded)
    decoded = Dense(2048, activation = 'relu')(decoded)
    decoded = Dense(df.shape[1:][0], activation ='sigmoid')(decoded) # 6 again number of features and should match input_data



    # this model maps an input to its reconstruction
    autoencoder = Model(input_data, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_data, encoded)

    # model optimizer and loss
    autoencoder = Model(input_data, decoded)

    # loss function and optimizer
    autoencoder.compile(optimizer='adam', loss='mse')

    # train test split
    from sklearn.model_selection import train_test_split
    x_train, x_test, = train_test_split(df, test_size=0.1, random_state=42)


    # train the model
    autoencoder.fit(df,
                  df,
                  epochs=20,
                  batch_size=8,
                  shuffle=True, verbose=0)

    #autoencoder.summary()

    # predict after training
    # note that we take them from the *test* set
    encoded_data = encoder.predict(df)
    encoded_data_test = encoder.predict(df_test)
    df_train_new=pd.DataFrame(encoded_data)
    df_test_new=pd.DataFrame(encoded_data_test)
    return(df_train_new,df_test_new)


def dimension_reduction(df_train,df_test,components, algo):
    if algo == 'PCA':
        pca = PCA(n_components=components)
        df_train = pca.fit_transform(df_train)
        df_test = pca.transform(df_test)
    elif algo == 'KPCA':
        kpca = KernelPCA(kernel = 'rbf', n_components=components)
        df_train = kpca.fit_transform(df_train)
        df_test = kpca.transform(df_test)
    elif algo == 'TSNE':
        tsne = TSNE(method='exact',  n_components=components)
        df_train = tsne.fit_transform(df_train)
        df_test = tsne.fit_transform(df_test)
    elif algo == 'UMAP':
        umaps = UMAP(n_components=components)
        df_train = umaps.fit_transform(df_train)
        df_test = umaps.transform(df_test)
    elif algo == 'TSVD':
        svd = TruncatedSVD(n_components=components)
        df_train = svd.fit_transform(df_train)
        df_test = svd.transform(df_test)
    elif algo == 'ISO':
        isomap = Isomap(n_components=components)
        df_train = isomap.fit_transform(df_train)
        df_test = isomap.transform(df_test)
    elif algo == 'AE' :
        df_train, df_test = auto_encoder(df_train,df_test, components)
        #df_test = auto_encoder(df_test, components)
    else:
        print('Data looks good enough, NO High Dimensionality')
        
    return df_train,df_test
#Dimensionality reduction
def optimal_components(df_train, df_test):
    pca = PCA() 
    X_train = pca.fit_transform(df_train)
    X_test = pca.transform(df_test)
    total=sum(pca.explained_variance_)
    k=0
    current_variance=0
    while current_variance/total < 0.95: # Change this to see increased accuracy in majority models, I tried with 98 as well but nof components will be 20+ in that case
        current_variance += pca.explained_variance_[k]
        k=k+1
    return k
# Testing the logic
k =  optimal_components(train, test)
#No of components for dimension Reduction

#Apply PCA with n_componenets
pca = PCA(n_components=k)
x_train = pca.fit_transform(train)
x_test = pca.transform(test)
plt.figure(figsize=(20,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Exoplanet Dataset Explained Variance')
plt.show()
# SYNTHETIC SAMPLING
# GENERATE SYNTHETIC BUT DIFF SAMPLE(OVERSAMPLE THE MINORITY CLASS)
def label_balance(df_train, df_test):
    balancing = SMOTE(random_state=42, sampling_strategy='minority')
    X_train_res, Y_train_res = balancing.fit_resample(df_train, df_test)
    return X_train_res, Y_train_res
# Trying the model for one time just to check
X_train, X_test, Y_train, Y_test = train.iloc[:,1:], test.iloc[:,1:],train.iloc[:,0], test.iloc[:,0]

# Step 2: Normalizing the data
X_train,X_test, scaling = data_normalizing(X_train,X_test,4)

#Step 3.0 finding optimal components
k = optimal_components(X_train, X_test)

#Step 3: Dimension reduction in this case, very high dimesnions
#print(algo)
X_train,X_test = dimension_reduction(X_train,X_test, k, 'PCA')

#Step 4: Imbalanced class, dominant class
#print(Y_train.value_counts())
X_train, Y_train = label_balance(X_train,Y_train)
rf = RandomForestClassifier(random_state=1,n_jobs =-1)
rf.fit(X_train, Y_train)
Y_predict = rf.predict(X_test)
scores = cross_validate(rf, X_train, Y_train, cv=5 ,scoring=['accuracy','f1', 'precision','recall']  )

print(scores['test_accuracy'].mean(),
      scores['test_accuracy'].std(),
        scores['test_f1'].mean(),
        scores['test_precision'].mean(),
        scores['test_recall'].mean())
# Building Model
def ml_model(classifier, X_train, X_test, Y_train, Y_test):
    classifier.fit(X_train, Y_train)
    Y_predict = classifier.predict(X_test)
    scores = cross_validate(classifier,X_train, Y_train, cv=5, scoring=['accuracy','f1', 'precision','recall'] )
    cv_accuracy = scores['test_accuracy']
    acc_mean  = scores['test_accuracy'].mean()
    acc_variance  = scores['test_accuracy'].std()
    f1_mean = scores['test_f1'].mean()
    precision_mean = scores['test_precision'].mean()
    recall_mean = scores['test_recall'].mean()
    #print("Accuracy mean: "+ str(cv_mean))
    #print("Accuracy variance: "+ str(cv_variance))
    acc = accuracy_score(Y_test, Y_predict)
    conf_matrix = confusion_matrix(Y_test, Y_predict)
    return acc, conf_matrix, cv_accuracy, acc_mean, acc_variance, f1_mean, precision_mean, recall_mean
def data_scaling(df_train,df_test,option):
    #Step1: Memory management of dataset, changing type by downcasting it so that it occupies less space
    df_train = memory_manager(df_train)
    df_test = memory_manager(df_test)
    X_train, X_test, Y_train, Y_test = df_train.iloc[:,1:], df_test.iloc[:,1:],df_train.iloc[:,0], df_test.iloc[:,0]
    
    # Step 2: Normalizing the data
    X_train,X_test, scaling = data_normalizing(X_train,X_test,option)
    
    # Step check : we check for optimal no of component using PCA for every scaling type
    k = optimal_components(X_train, X_test)
    return X_train,X_test,Y_train,Y_test, scaling ,k


def data_prep(df_train,df_test,label_train, label_test, scale_type,dim_red_ago,components):
    
    #Step 3: Dimension reduction in this case, very high dimesnions
    #print(algo)
    df_train_reduced,df_test_reduced = dimension_reduction(df_train,df_test, components, dim_red_ago)
    
    #Step 4: Imbalanced class, dominant class
    #print(Y_train.value_counts())
    print
    X_train, Y_train = label_balance( df_train_reduced,label_train)
    X_test, Y_test = df_test_reduced, label_test
    #print(Y_train.value_counts())
    
    pipeline_optim =  {'Scaling Algorithm': scale_type, 'Dimensional Reduction': dim_red_ago}
    
    return X_train, X_test, Y_train, Y_test, pipeline_optim


def result(df_train, df_test, scaling, dimension_algo):
    # Identifying the train ad
    #X_train, X_test, Y_train, Y_test = df_train.iloc[:,1:], df_test.iloc[:,1:],df_train.iloc[:,0], df_test.iloc[:,0]
    X_train,X_test,Y_train,Y_test, scaling ,k = data_scaling(df_train,df_test,scaling)
    X_train, X_test, Y_train, Y_test, pipes = data_prep(X_train,X_test,Y_train, Y_test, scaling, dimension_algo ,k)
    RFC = RandomForestClassifier(random_state = 7, n_jobs =-1)
    acc, conf_matrix, cv_accuracy, acc_mean, acc_variance, f1_mean, precision_mean, recall_mean = ml_model(RFC,X_train, X_test, Y_train, Y_test)
    #pipes['result'] = acc*100
    pipes['Dimensional Components'] = k
    pipes['Accuracy CV'] = cv_accuracy*100
    #pipes['cv accuracy'] = [round(val,3) for val in pipes['cv accuracy']]
    pipes['Mean Accuracy CV'] = acc_mean*100
    pipes['Variance Accuracy CV'] = acc_variance*100
    pipes['Mean F1 CV'] = f1_mean*100
    pipes['Mean Precision CV'] = precision_mean*100
    pipes['Mean Recall CV'] = recall_mean*100
    return pipes   
# Testing the model for single instance for Standard Scaler and PCA for optmial component
my_result = result(train, test, 1 , 'TSVD')
my_result
def final(X_train, X_test, Y_train, Y_test, scaling, dimension_algo,k):
    X_train, X_test, Y_train, Y_test, pipes = data_prep(X_train,X_test,Y_train, Y_test, scaling, dimension_algo ,k)
    RFC = RandomForestClassifier(random_state = 7)
    acc, conf_matrix, cv_accuracy, acc_mean, acc_variance, f1_mean, precision_mean, recall_mean = ml_model(RFC,X_train, X_test, Y_train, Y_test)
    #pipes['result'] = acc*100
    pipes['Dimensional Components'] = k
    pipes['Accuracy CV'] = cv_accuracy*100
    #pipes['cv accuracy'] = [round(val,3) for val in pipes['cv accuracy']]
    pipes['Mean Accuracy CV'] = acc_mean*100
    pipes['Variance Accuracy CV'] = acc_variance*100
    pipes['Mean F1 CV'] = f1_mean*100
    pipes['Mean Precision CV'] = precision_mean*100
    pipes['Mean Recall CV'] = recall_mean*100
    return pipes   

#==============================================================================================================================
#    FINAL MODEL WHICH WILL RUN OUR MODEL RANDOM FOREST FOR ALL COMBINATIONS OF SCALING AND DIMENSION REDUCTION TECHNIQUES
#==============================================================================================================================

scaling_options = {1: 'Standard Scalar',2:'MinMax Scalar',3:'MaxAbs Scalar',4:'Normalizer',5:'Robust Scaler'}
Dimension_red_options = ['PCA','KPCA','UMAP','TSVD','ISO']

list_result = []

for scale in scaling_options:
    X_train,X_test,Y_train, Y_test, scaling ,k = data_scaling(train,test,scale)
    for dim in Dimension_red_options:
        #start_time = time.time()
        res = final(X_train, X_test, Y_train, Y_test, scaling, dim,k)
        #end_time = time.time()
        print(res)
        list_result.append(res) 
Results = pd.DataFrame.from_dict(list_result, orient='columns')
Results
def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: darkgreen' if is_max.any() else '' for v in is_max]

def highlight_lesserthan(s, threshold, column):
    is_min = pd.Series(data=False, index=s.index)
    is_min[column] = s.loc[column] <= threshold
    return ['background-color: darkorange' if is_min.any() else '' for v in is_min]

print("\x1b[32m\" Cross Validation Accuracy Mean in Dark Green\"\x1b[0m")
print("\x1b[33m\" Cross Validation Accuracy Variance in color Dark Orange\"\x1b[0m")
Results.style.\
         apply(highlight_greaterthan, threshold=max(Results['Mean Accuracy CV']), column=['Mean Accuracy CV'], axis=1).\
         apply(highlight_lesserthan, threshold=min(Results['Variance Accuracy CV']), column=['Variance Accuracy CV'], axis=1)
box = pd.DataFrame({'preprocess': Results['Scaling Algorithm'] + '-->' + Results['Dimensional Reduction'] , 'res': Results['Accuracy CV'] })
boxt = box.T
boxt = boxt.reset_index(drop = True)
boxt.columns = boxt.iloc[0]
boxt = boxt.iloc[1:]
boxt = boxt.apply(pd.Series.explode).reset_index(drop = True)
boxt.astype('float64').dtypes
import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(20,15) )
sns.boxplot(y=0, x="value", data=pd.melt(boxt), orient="h" , showfliers = False )
ax.set_title('Box Plot of accuracies for different models')
ax.set_xlabel('Accuracy')
ax.set_ylabel('Model')
ax.grid(True)
plt.show()
#replicate the best result
#scaling_options = {1: 'Standard Scalar',2:'MinMax Scalar',3:'MaxAbs Scalar',4:'Normalizer',5:'Robust Scaler'}
#Dimension_red_options = ['PCA','KPCA','UMAP','TSVD','ISO']
Best = result(train, test, 4 , 'PCA')
Best
# Notes:
# We need to understand the tradeoff between accuracy and Computation cost. 
# Also when we are doing PCA variance check to find the best no of components, Lot depends on cutoff value as well
# comments are welcomed
