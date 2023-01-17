# imports 



import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from functools import partial

from sklearn.feature_selection import mutual_info_classif, SelectKBest

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report

from sklearn.base import BaseEstimator, TransformerMixin

from functools import partial

from sklearn.model_selection import GridSearchCV



from IPython.display import display

import time



import warnings

pd.options.display.max_colwidth = 200

warnings.filterwarnings(action='ignore')
# Load data 

data = pd.read_csv(r'../input/ckdisease/kidney_disease.csv')

# Split it to train and test

train_data, test_data = train_test_split(data) 

train_data = train_data.set_index('id')

test_data = test_data.set_index('id')
train_data
# data information

train_data.info()
#Get the shape of the data (the number of rows & columns)

print("Shape :",train_data.shape)

# Statistical description of training data set

print(train_data.describe())
# Describe columns not taken into consideration in cell above, we will deal with categorical data after cleaning the data

train_data[[c for c in train_data.columns if c not in train_data.describe().columns]].astype('category').describe()
train_data['cad'].unique()
def removeNonAlphanumeric(df) :

    """ 

    Remove non-alphanumeric characters from data values

    Input :

        df -- dataframe 

    Output :

        df -- cleaned dataframe

    """

    for c in df.columns :

        if df[c].dtype == "O" :

            df[c] = df[c].str.replace('\t', '')

            df[c] = df[c].str.replace('[^a-zA-Z0-9]', '')

    df = df.replace('',np.nan)

    return df



def toNumeric(df):

    """" 

    Convert string column corresponding to numerical values to numerical columns

    Input : 

        df -- dataframe 

    Output :

        df -- dataframe with converted columns

    """

    for c in df.columns :

        if df[c].dtype == "O" and all(df[c].str.isnumeric()):

            df[c] = pd.to_numeric(df[c])

    return df

            





class HandleMissingTransformer(BaseEstimator, TransformerMixin):

    """Customized transformer to handles missing data"""

    

    def __init__(self, method,constant = ''):

        '''' 

        Initialise The transformer

        Inputs :

            method -- method used to replace or impute missing data (drop/constant/most_frequent/median/mean)

            constant -- if constant method is selected, the value of the constant must be specified

        '''

        self.method = method

        self.constant = constant

        self.imputerDict = {}

        



    def fit(self, df ):

        '''

        If impute method is selected i.e self.method not in ['drop', 'constant'], we must fit an imputer

        Input : 

            df -- data with missing

        '''

        if self.method not in ['drop', 'constant'] :

            if self.method != "most_frequent":

                print("For non numerical columns, most frequent strategy is used")

            for c in df.columns :

                imp = SimpleImputer(missing_values=np.nan, strategy=self.method if df[c].dtype!="O" else "most_frequent")

                imp = imp.fit(df[[c]])

                self.imputerDict[c] = imp 

        return self

            

                

        

    def transform(self, df):

        """

        If impute method is selected, impute missing values using imput_dict created in fit function

        Input : 

            df -- data with missing values

        """

        if self.method == "drop" :

            df = df.dropna(inplace= True)

        elif self.method == 'constant' :

            df.fillna(self.constant, inplace= True)

        else :

            for c in df.columns : 

                df[c] = self.imputerDict[c].transform(df[[c]])

        return df  

    

def getCategFeat(df, n, target):

    """

    get dataframe's categorical features 

    Inputs :

        df     -- dataframe  

        n      -- min modalities for numerical features

        target -- target column name

    """

    return [c for c in df.columns if (df[c].dtype == 'O' or df[c].nunique()<n) and c!=target]
# Remove non alphanumeric 

transf_alphaN = FunctionTransformer(removeNonAlphanumeric, validate= False) 

# Transform function to Transformer object so that we can use in pipeline at test time

transf_num = FunctionTransformer(toNumeric, validate= False)

train_data = transf_alphaN.transform(train_data)

train_data = transf_num.transform(train_data)

train_data['cad'].unique()
# Get columns with null values

print("Columns with null values before imputing")

print(train_data.columns[train_data.isna().any()].tolist())



# Handle missing values

transf_Missing = HandleMissingTransformer(method="most_frequent")

train_data = transf_Missing.fit(train_data).transform(train_data)



print("Columns with null values after imputing")

print(train_data.columns[train_data.isna().any()].tolist())
minModalities = 7 ## To match dataset desciprtion on UCI website, numerical variables with less than 7 modalities are considered as nominal

categorical_features = getCategFeat(train_data,minModalities,'classification')

print("Categorical Features : ",categorical_features)

numerical_features = [c for c in train_data.columns if c not in categorical_features and c!="classification"]

print("Numerical Features : ", numerical_features)
def target_variable_exploration(df, target, xlabel, ylabel, title, positive=1) :

    """ 

    plots the distribution of the classes

    Input :

        df -- dataframe containing classes

        target -- class column

        xlabel

        ylabel 

        title

        positive -- modality corresponding to positive class

    """

    negative =  [c for c in df[target].unique() if c !=positive][0]

    positive_class = df[target].value_counts()[positive]

    negative_class = df[target].shape[0] - positive_class

    positive_per = positive_class / df.shape[0] * 100

    negative_per = negative_class / df.shape[0] * 100

    plt.figure(figsize=(8, 8))

    sns.countplot(df[target], order=[positive, negative]);

    plt.xlabel(xlabel, size=15, labelpad=15)

    plt.ylabel(ylabel, size=15, labelpad=15)

    plt.xticks((0, 1), [ 'Positive class ({0:.2f}%)'.format(positive_per), 'Negative class ({0:.2f}%)'.format(negative_per)])

    plt.tick_params(axis='x', labelsize=13)

    plt.tick_params(axis='y', labelsize=13)

    plt.title(title, size=15, y=1.05)

    plt.show()
### Target variable exploration



target_variable_exploration(train_data, "classification", 'Class', ' Count', 'Training Set ckd Distribution', positive = 'ckd')
# Pairplots

sns.pairplot(train_data, diag_kind ='hist' , hue="classification")

plt.show()
def plot_numeric(data, numeric_features, target) :

    """ 

    plots analysing numerical features

    Inputs : 

        data -- dataframe containing features to plot

        numeric_features -- list of numerical features

        target -- target column name

     """

    # Looping through and Plotting Numeric features

    for column in numeric_features:    

        # Figure initiation

        fig = plt.figure(figsize=(18,12))



        ### Distribution plot

        sns.distplot(data[column], ax=plt.subplot(221));

        # X-axis Label

        plt.xlabel(column, fontsize=14);

        # Y-axis Label

        plt.ylabel('Density', fontsize=14);

        # Adding Super Title (One for a whole figure)

        plt.suptitle('Plots for '+column, fontsize=18);



        ### Distribution per Positive / Negative class Value

        # Not Survived hist

        classes = data[target].unique()

        sns.distplot(data.loc[data[target]==classes[0], column].dropna(),

                     color='red', label=str(classes[0]), ax=plt.subplot(222));

        # Survived hist

        sns.distplot(data.loc[data[target]==classes[1], column].dropna(),

                     color='blue', label=str(classes[1]), ax=plt.subplot(222));

        # Adding Legend

        plt.legend(loc='best')

        # X-axis Label

        plt.xlabel(column, fontsize=14);

        # Y-axis Label

        plt.ylabel('Density per '+ str(classes[0])+' / '+str(classes[1]), fontsize=14);



        ### Average Column value per positive / Negative Value

        sns.barplot(x=target, y=column, data=data, ax=plt.subplot(223));

        # X-axis Label

        plt.xlabel('Positive or Negative?', fontsize=14);

        # Y-axis Label

        plt.ylabel('Average ' + column, fontsize=14);



        ### Boxplot of Column per Positive / Negative class Value

        sns.boxplot(x=target, y=column, data=data, ax=plt.subplot(224));

        # X-axis Label

        plt.xlabel('Positive or Negative ?', fontsize=14);

        # Y-axis Label

        plt.ylabel(column, fontsize=14);

        # Printing Chart

        plt.show()

        

def plot_categ(train_data, target, nominal_features,positive =1) :

    """ 

    plots analysing nominal categorical features

    Inputs : 

        data -- dataframe containing features to plot

        nominal_features -- list of nominal features

        target -- target column name

     """

    # Looping through and Plotting Categorical features

    for column in nominal_features:

    # Figure initiation

        fig = plt.figure(figsize=(18,12))

        

        ### Number of occurrences per categoty - target pair

        ax = sns.countplot(x=column, hue=target, data=train_data, ax = plt.subplot(211));

        # X-axis Label

        plt.xlabel(column, fontsize=14);

        # Y-axis Label

        plt.ylabel('Number of occurrences', fontsize=14);

        # Adding Super Title (One for a whole figure)

        plt.suptitle('Plots for '+column, fontsize=18);

        # Setting Legend location 

        plt.legend(loc=1);



        ### Adding percents over bars

        # Getting heights of our bars

        height = [p.get_height() if np.isnan(p.get_height()) == 0 else 0 for p in ax.patches] #  get nan if

        # Counting number of bar groups 

        ncol = int(len(height)/2)

        # Counting total height of groups

        total = [height[i] + height[i + ncol] for i in range(ncol)] * 2

        # Looping through bars

        for i, p in enumerate(ax.patches):    

            # Adding percentages

            ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 10,

                    '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14) 



        negative = train_data[target].unique()[0] if train_data[target].unique()[0] != positive else train_data[target].unique()[1]

        ### Positive class percentage for every value of feature

        

        sns.pointplot(x=train_data[column], y=train_data[target].map({negative:0 , positive: 1}), ax = plt.subplot(212));

        # X-axis Label

        plt.xlabel(column, fontsize=14);

        # Y-axis Label

        plt.ylabel(' Positive class percentage', fontsize=14);

        # Printing Chart

        plt.show()



### Plotting Numeric Features

plot_numeric(train_data, numerical_features, 'classification')

##### Plotting nominal Categorical Features    



plot_categ(train_data,"classification", categorical_features[:-2],positive='ckd')
def correlationMap(df, target) :

    """ 

    Correlation Heatmap

    Inputs : 

        df -- dataframe containing features to plot

        target -- target column name

     """

    classes = df[target].unique()

    if data[target].dtype == 'O' :

        df[target+'_id'] = (df[target]== classes[0]).astype(int) #encode string target 

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))

    sns.heatmap(corr, vmax=.8,annot=True, square=True)

    if data[target].dtype == 'O' :

        df.drop([target+'_id'], axis=1, inplace=True)

    # fix for matplotlib bug that cuts off top/bottom of seaborn viz

    b, t = plt.ylim() # Gets the values for bottom and top

    b += 0.5 # Add 0.5 to the bottom

    t -= 0.5 # Subtract 0.5 from the top

    plt.ylim(b, t) # update the ylim(bottom, top) values

    plt.show()
# Correlation Analyses



correlationMap(train_data,'classification')
# We can delete highly correlated features because they can negatively impact some models (eg. trees)

train_data, test_data = train_data.drop("pcv", axis=1), test_data.drop("pcv", axis=1)

numerical_features = [f for f in numerical_features if f!="pcv"]

numerical_features
def featureEng(numerical_features, categorical_features):

    """ 

    create pipeline for feature preprocessing 

    Inputs : 

        numerical_features -- list of numerical features

        categorical_features -- list of categorical features

    Outputs :

        preproc -- pipeline with feature preprocessing steps

     """

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    t =  ColumnTransformer([('Scaler', numeric_transformer, numerical_features),('OneHotEncod', categorical_transformer, categorical_features)])

    preproc = Pipeline(steps=[('preprocessor', t)])

    return preproc



# Encoding categorical features



transf_train = featureEng(numerical_features, categorical_features).fit(train_data)

X_train = transf_train.transform(train_data)

y_train = train_data['classification'].replace({'ckd':1,'notckd':0}).values
transf_train
#  get columns names after transformations

columns_eng = numerical_features + transf_train.named_steps['preprocessor'].transformers_[1][1].get_feature_names(categorical_features).tolist()
# PCA on numerical features



pca = PCA(n_components=10)

principalComponents = pca.fit_transform(X_train[:,:len(numerical_features)])



# Keep two PC

principalDf = pd.DataFrame(data = principalComponents[:,:2]

             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, pd.DataFrame(y_train)], axis = 1)
pca.explained_variance_
pca.explained_variance_ratio_
fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1]

colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf[0] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
# Projection on first PC

n_axes = len(numerical_features)

_, axes = plt.subplots(ncols=5,nrows=2, figsize=(20,10))

col_id = 0

for i in range(axes.shape[0]):

    for j in range(axes.shape[1]):

        try :

            axes[i][j].scatter(principalComponents[:,0], X_train[:,col_id])

            axes[i][j].set_title(f'1st component vs {numerical_features[col_id]}')

            col_id = col_id+1

        except exception as e:

            print(e)

            break
# # Projection on second PC

n_axes = len(numerical_features)

_, axes = plt.subplots(ncols=5,nrows=2, figsize=(20,10))

col_id = 0

for i in range(axes.shape[0]):

    for j in range(axes.shape[1]):

        try :

            axes[i][j].scatter(principalComponents[:,1], X_train[:,col_id])

            axes[i][j].set_title(f'2st component vs {columns_eng[col_id]}')

            col_id = col_id+1

        except :

            break
components = pca.components_

plt.figure(figsize=(10,10))

for i, (x, y) in enumerate(zip(components[0,:], components[1,:])):

    plt.plot([0, x], [0, y], color='k')

    plt.text(x, y, numerical_features[i])



plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')

plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')



plt.xlim(-0.7, 0.7)

plt.ylim(-0.7, 0.7)
X_train.shape


ndiscrete = len(transf_train.named_steps['preprocessor'].transformers_[1][1].get_feature_names(categorical_features))

discreteCol = np.arange(len(X_train[0])-ndiscrete,len(X_train[0]))
class selectFeaturesTransformer(BaseEstimator, TransformerMixin):

    """Custom scaling transformer"""

    def __init__(self, k=10,method='RF',discreteCol=[]):

        """ 

        initialize transformer

        Inputs : 

            k -- number of features to keep

            method -- method to use, either 'Mutual Information or RF

            discreteCol -- if Mutual Information is used, specify indexes of discrete columns

        """

        self.k = k

        self.method = method

        self.order = []

        self.discreteCol = discreteCol

        

        

        



    def fit(self, X_train,y_train):

        """

        Fit the transformer on data

        Input :

            X_train -- features array

            Y_train -- labels array

        Output :

            fitted transformer

        """

        if self.method == "Mutual Information" :

            discrete_mutual_info_classif = partial(mutual_info_classif, 

                                                   discrete_features=self.discreteCol)

            featS = SelectKBest(k=self.k, score_func=discrete_mutual_info_classif).fit(X_train,y_train )

            self.order = np.flip(featS.scores_.argsort())

            #self.selectedColumns = [columns_eng[i]  for i in self.order[:self.k]]

            #return X_train[:,order_mi[:self.k]]

        

        else :

            rfModel = RandomForestClassifier(random_state =0).fit(X_train, y_train)

            order = np.flip(rfModel.feature_importances_.argsort())

            self.order = np.flip(rfModel.feature_importances_.argsort())

            #self.selectedColumns = [columns_eng[i]  for i in order_rf[:self.k]]

            #return X_train[:,order_[:self.k]]

        return self

            

                

        

    def transform(self, X_train):

        """

        apply fitted transformer to select features

        Input :

            X_train -- features array

        Output :

            array containing only selected features

        """

        return X_train[:,self.order[:self.k]]
FSelector_mi = selectFeaturesTransformer(k=10,method="Mutual Information",discreteCol=discreteCol)

FSelector_rf = selectFeaturesTransformer(k=10,method="Random Forest")

FSelector_mi.fit(X_train,y_train)

FSelector_rf.fit(X_train,y_train)
print("Top 10 selected by Mutual information")

print([columns_eng[i]  for i in FSelector_mi.order[:10]])

print("Top 10 selected by Random Forest")

print([columns_eng[i]  for i in FSelector_rf.order[:10]])
np.random.seed(44)


classifiers = [

    SGDClassifier(loss='log'), # for logistic regression

   KNeighborsClassifier(),

    SVC(),

    GaussianProcessClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier(),

    MLPClassifier(),

    GaussianNB()]



ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")



ker_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)



#ker_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))



kernel_list = [ker_rbf, ker_rq]



names = ["Logistic Regression with SGD", "Nearest Neighbors", "SVM", "Gaussian Process",

         "Decision Tree", "Random Forest","Gradient Boosting", "Neural Net",

         "Naive Bayes"]



parameters = {"Logistic Regression with SGD" : {'Classifier__penalty':['l1','l2',None],

                                               'Classifier__learning_rate' : ['constant','optimal','adaptive'],

                                               'Classifier__eta0' : [0.1]},

    "Nearest Neighbors" : {'Classifier__n_neighbors':[5,8,10]},

        'SVM':{'Classifier__kernel':['linear','rbf'],'Classifier__C':[0.1,0.5,1.,1.5]},

        "Gaussian Process":{"Classifier__kernel": kernel_list,

                            "Classifier__n_restarts_optimizer": [1, 2, 3]},

        "Decision Tree" : {"Classifier__max_features" : ['sqrt','log2',None],

                        "Classifier__max_depth":[10,30,50,None]}

        ,"Random Forest":{"Classifier__n_estimators":[8,10,20,50],"Classifier__max_features" : ['sqrt','log2',None],

                        "Classifier__max_depth":[10,30,50,None]},

       'Gradient Boosting':{"Classifier__max_features" : ['sqrt','log2',None],

                        "Classifier__max_depth":[2,3,10],

                        "Classifier__learning_rate":[1e-1,1e-2,1e-3]},

         "Neural Net" : {'Classifier__hidden_layer_sizes': [(20,20,20), (25,50,25), (50,50)],

                      'Classifier__activation': ['tanh', 'relu'],"Classifier__learning_rate_init":[1e-1,1e-2,1e-3]},

        "Naive Bayes" : {"Classifier__var_smoothing" : [1e-8, 1e-9]}

         }

parameters_featuresSelection = {'FeatureSelection__method':['RF'],'FeatureSelection__k':[10,20,30,47]}
def train(X_train, y_train, classifiers, names,parameters, parameters_featuresSelection, crossVal = True):

    """ 

    training process

    Inputs : 

        X_train -- features array

        Y_train -- labels array

        classifiers -- list of classifiers to test

        names -- list of classifiers names

        parameters -- tuning parameters corresponding for classifiers

        parameters_featuresSelection -- parameters for fearures selection

        crossVal -- whether to use cross validation or not

     """

    results = pd.DataFrame()

    for name, clf in zip(names, classifiers):

        print('############# ', name, ' #############')

        start = time.time()

        #print(params[name])

        FSelector = selectFeaturesTransformer()

        pipeline = Pipeline([('FeatureSelection',FSelector),('Classifier',clf)])

        parameters[name]['FeatureSelection__method'] = parameters_featuresSelection['FeatureSelection__method']

        parameters[name]['FeatureSelection__k']= parameters_featuresSelection['FeatureSelection__k']

        if crossVal:

            classifier = GridSearchCV(pipeline, parameters[name], cv=3)

        else:

            classifier = pipeline

        #print(classifier)

        classifier.fit(X_train, y_train)

        # All results

        means = classifier.cv_results_['mean_test_score']

        stds = classifier.cv_results_['std_test_score']

        r = pd.DataFrame(means,columns = ['mean_test_score'])

        r['std_test_score'] = stds

        r['params'] = classifier.cv_results_['params']

        r['classifier'] = name

        

        print('Training time (Cross Validation = ',crossVal,') :',(time.time()-start)/len(means))

        display(r.sort_values(by=['mean_test_score','std_test_score'],ascending =False))

        results = pd.concat([results, r], ignore_index=True)

        #for mean, std, params in zip(means, stds, classifier.cv_results_['params']):

        #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    results_sorted = results.sort_values(by=['mean_test_score','std_test_score'],ascending =False)

    return results_sorted
results = train(X_train, y_train, classifiers, names,parameters, parameters_featuresSelection)    
results.groupby('classifier').head(3)
# apply transformations on test data

test_data = transf_alphaN.transform(test_data)

test_data = transf_num.transform(test_data)

test_data = transf_Missing.transform(test_data)

y_test = test_data['classification'].replace({'ckd':1,'notckd':0}).values

X_test = transf_train.transform(test_data)
X_test
model_selected = results.iloc[0]

model = classifiers[names.index(model_selected['classifier'])]

param = {key.split('__')[1]:val for key,val in model_selected['params'].items() if 'FeatureSelection' not in key } 

model.set_params(**param)
model.fit(X_train,y_train)

model.score(X_test, y_test)
print(classification_report(y_test, model.predict(X_test), target_names=['notckd','ckd']))