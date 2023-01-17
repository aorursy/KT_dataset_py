!pip install seaborn --upgrade #Update Seaborn for Plotting
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno #Visualize null



#Plotting Functions

import matplotlib.pyplot as plt



#Aesthetics

import seaborn as sns

sns.set_style('ticks') #No grid with ticks

print(sns.__version__)
#Data Import

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
income=pd.read_csv('/kaggle/input/income/train.csv')

income.head()
income.info()

msno.bar(income)
#Generating column labels in data frame for simple copy paste

income.columns
#Using a for loop we can print each unique value per column.

col_list=['workclass','education','marital-status', 'occupation','relationship', 'race', 'gender','native-country']

for label in col_list:

    print('***',label, income[label].unique()) #*** are used for indentation
new_train=income.copy()#Setting a check point for errors. This is good practice for when manipulating values on data frames through multiple code blocks



#Renaming income for simple input

new_train.rename(columns={'income_>50K':'income'}, inplace=True)



#workclass

new_train['workclass'].fillna('nan', inplace=True)#First replace the null value with a 'nan' as part of that column contains them already. The .fillna function replaces nulls with whatever string label you want

new_train['workclass']=new_train['workclass'].replace({'Self-emp-not-inc':'Self-emp','Self-emp-inc':'Self-emp',

        'Never-worked':'Un-emp/Unk','Without-pay':'Un-emp/Unk', 'State-gov':'Government', 

        'Federal-gov':'Government','Local-gov':'Government','nan':'Un-emp/Unk'}) #This first line is a harsh way of doing this and is very tedious.



"""Looking below we can abbreviate this through several for loops using the replace function. Setting a list of 

variables that you want to converge to a single variable as seen below. Once a list is made you can implement 

it into the for loop and place the pulled values to be replaced by a specific value (pull label:new label)"""



#Education

edu_lHS=['12th','7th-8th','9th','10th', '11th','5th-6th','1st-4th','Preschool'] # <HS

edu_assoc=['Assoc-voc','Assoc-acdm','Some-college'] #Associate

edu_mpro=['Masters','Prof-school']#Mas-Pro

for edu in edu_lHS:

    new_train['education']=new_train['education'].replace({edu:'<HS'})

for edu in edu_assoc:

    new_train['education']=new_train['education'].replace({edu:'Associate'})

for edu in edu_mpro:

    new_train['education']=new_train['education'].replace({edu:'Mas-Pro'})



#Marital Status

sep_div=['Divorced','Separated']#Sep-Div

married=['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse']#Married

for mar in sep_div:

    new_train['marital-status']=new_train['marital-status'].replace({mar:'Sep-Div'})

for mar in married:

    new_train['marital-status']=new_train['marital-status'].replace({mar:'Married'})



#Occupation

new_train['occupation'].fillna('nan', inplace=True)



#Relationship: 

marr=['Husband', 'Wife']#Married

for mar in marr:

    new_train['relationship']=new_train['relationship'].replace({mar:'Married'})



#Native Country

europe=['England', 'Italty', 'Germany', 'France','Yugoslavia', 'Poland', 'Greece', 'Ireland', 'Scotland',

       'Hungary','Holand-Netherlands','Portugal']

asia=['China', 'Philippines','Vietnam','Thailand','Taiwan','Laos','Cambodia','Japan', 'Hong','India','Iran']

caribbean=['Jamaica','Dominican-Republic','Cuba','Haiti','Trinadad&Tobago', 'Puerto-Rico']

n_america=['United-States','Canada']

c_america=['Mexico','Honduras','El-Salvador','Guatemala','Nicaragua']

s_america=['Columbia','Ecuador','Peru']

for coun in europe:

    new_train['native-country']=new_train['native-country'].replace({coun:'Europe'})

for coun in asia:

    new_train['native-country']=new_train['native-country'].replace({coun:'Asia'})

for coun in caribbean:

    new_train['native-country']=new_train['native-country'].replace({coun:'Carribean'})

for coun in n_america:

    new_train['native-country']=new_train['native-country'].replace({coun:'N.America'})

for coun in c_america:

    new_train['native-country']=new_train['native-country'].replace({coun:'C.America'})

for coun in s_america:

    new_train['native-country']=new_train['native-country'].replace({coun:'S.America'})

new_train['native-country'].fillna('nan', inplace=True)

new_train['native-country']=new_train['native-country'].replace({'South':'nan'})#As we do not know what south is we will just convert it to nan



#Once we are done we can look that we replaced all the null values for either 'nan' or something else

new_income=new_train.copy()# Train was a fragment of a previous notebook I was working on and did not want to replace it for every line.

new_income.info()

for label in col_list:

    print(label, new_income[label].unique())

new_income.head()
def cleaner(data,column, group_by=False, category=None, score=None,  

            old_label1=None, new_label1=None, old_label2=None , new_label2=None, 

            old_label3=None, new_label3=None, old_label4=None, new_label4=None, 

            old_label5=None, new_label5=None, old_label6=None,new_label6=None):

    

    """ Quick helper function to make data cleaning faster and require less lines 

    of code per column for replacing values. Following the code you will see that 

    old_label#=old_label#. Example (1:1 as 2:2.). This function will automatically

    fill null values as 'nan'. If you do not want 'nan' it must be specified in

    list.

    

    Inputs:

    data: pandas DataFrame

    column: 'column name' as string

    old_label: variable of a string or string list of values. default=None

    new_label: variable of a string to replace the old_label(s) with. default=None

    group_by: Groups Data by category, default=False, bool

    category: Category in DataFrame to group data by, default=None, string

    score: Transform null by string, default=None, 'mean', 'median', 'mode'

    ***Note group_by, category and score all must have a value to be used together***

    

    Output: Updated DataFrame in specified column and a print of the number of sets of strings replaced"""

    

    print('***',column,'***')

    #Null Filling

    if data[column].isnull().sum()==0:

        print('No null in', column)

    elif group_by==True:

        null=data[column].isnull().sum()

        data[column].fillna(data.groupby(category)[column].transform(score), inplace=True)

        print(column,'null filled by', score, 'of',category)

        print(null, 'null values filled in', column)

    else:

        null=data[column].isnull().sum()

        data[column].fillna('nan', inplace=True)

        print(null, 'null values filled in', column)

        

    #First set of labels

    if old_label1==None:

        return print('No label replacements made!')

    else:

        for old in old_label1:

            new_data=data[column].replace({old:new_label1}, inplace=True)

            x=1

    

    #Second set of labels

    if old_label2==None:

        return print(x, 'set of labels were replaced in',column)

    else:

        for old in old_label2:

            new_data=data[column].replace({old:new_label2}, inplace=True)

            x2=x+1

    

    #Third set of labels        

    if old_label3==None:

        return print(x2, 'set of labels were replaced in', column)

    else:

        for old in old_label3:

            new_data=data[column].replace({old:new_label3}, inplace=True)

            x3=x2+1

    

    #Fourth set of labels

    if old_label4==None:

        return  print(x3, 'set of labels were replaced in', column)

    else:

        for old in old_label4:

            new_data=data[column].replace({old:new_label4}, inplace=True)

            x4=x3+1

            

    #Fifth set of labels

    if old_label5==None:

        return print(x4, 'set of labels were replaced in', column)

    else:

        for old in old_label5:

            new_data=data[column].replace({old:new_label5}, inplace=True)

            x5=x4+1

    

    #Sixth set of labels

    if old_label6==None:

        return print(x5, 'set of labels were replaced in', column)

    else:

        for old in old_label6:

            new_data=data[column].replace({old:new_label6}, inplace=True)

            x6=x5+1

            return print(x6, 'set of labels were replaced in', column)
new_income=income.copy()#Re-using new_income



#Renaming income for simple input

new_income.rename(columns={'income_>50K':'income'}, inplace=True)



#age

cleaner(new_income, 'age')



#workclass

work_self=['Self-emp-not-inc','Self-emp-inc']#Self-emp

work_unuk=['Never-worked','Without-pay','nan']#Un-emp/Unk

work_gov=['State-gov','Federal-gov','Local-gov']#Government

cleaner(new_income, 'workclass', work_self, 'Self-emp', work_unuk, 'Un-emp/Unk', work_gov, 'Government')



#Education

edu_lHS=['12th','7th-8th','9th','10th', '11th','5th-6th','1st-4th','Preschool'] # <HS

edu_assoc=['Assoc-voc','Assoc-acdm','Some-college'] #Associate

edu_mpro=['Masters','Prof-school']#Mas-Pro

cleaner(new_income, 'education', edu_lHS, '<HS', edu_assoc, 'Associate', edu_mpro, 'Mas-Pro')



#Occupation

new_income['occupation'].fillna('nan', inplace=True)#We only need to fill nan here



#Marital Status

sep_div=['Divorced','Separated']#Sep-Div

married=['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse']#Married

cleaner(new_income, 'marital-status', sep_div, 'Sep-Div', married, 'Married')



#Relationship

marr=['Husband', 'Wife']#Married

cleaner(new_income, 'relationship', marr, 'Married')



#Native Country

europe=['England', 'Italty', 'Germany', 'France','Yugoslavia', 'Poland', 'Greece', 'Ireland', 'Scotland',

       'Hungary','Holand-Netherlands','Portugal']

asia=['China', 'Philippines','Vietnam','Thailand','Taiwan','Laos','Cambodia','Japan', 'Hong','India','Iran']

caribbean=['Jamaica','Dominican-Republic','Cuba','Haiti','Trinadad&Tobago', 'Puerto-Rico']

n_america=['United-States','Canada']

c_america=['Mexico','Honduras','El-Salvador','Guatemala','Nicaragua']

s_america=['Columbia','Ecuador','Peru']

cleaner(new_income, 'native-country', europe, 'Europe', asia, 'Asia', caribbean, 'Caribbean', n_america,'N.America', c_america,'C.America', s_america,'S.America')

new_income['native-country']=new_income['native-country'].replace({'South':'nan'})#As we do not know what south is we will just convert it to nan



#Checking the data

new_income.info()

for label in col_list:

    print(label, new_income[label].unique())

new_income.head()
iris=pd.read_csv('/kaggle/input/iris/Iris.csv')

iris.info()

iris.head() 
cleaner(iris,'Species')

cleaner(iris,'SepalLengthCm')
fig, ax=plt.subplots()# Required for ax parameterization



sns.scatterplot(data=iris, x='SepalLengthCm' ,y='SepalWidthCm', hue='Species', palette=['b','r','g'], s=30)





#Ticks

ax.tick_params(direction='out', length=5, width=3, colors='k',

               grid_color='k', grid_alpha=1,grid_linewidth=2)

plt.xticks(fontsize=12, fontweight='bold', rotation=0)

plt.yticks(fontsize=12, fontweight='bold')





#Labels

plt.xlabel('Sepal Length (cm)', fontsize=12, fontweight=None, color='k')

plt.ylabel('Sepal Width (cm)', fontsize=12, fontweight=None, color='k')



#Removing Spines and setting up remianing

ax.spines['top'].set_color(None)

ax.spines['right'].set_color(None)

ax.spines['bottom'].set_color('k')

ax.spines['bottom'].set_linewidth(3)

ax.spines['left'].set_color('k')

ax.spines['left'].set_linewidth(3)



plt.savefig('Sepal.png')
def Plotter(plot, x_label, y_label, x_rot=None, y_rot=None,  fontsize=12, fontweight=None, legend=None, save=False,save_name=None):

    """

    Helper function to make a quick consistent plot with few easy changes for aesthetics.

    Input:

    plot: sns or matplot plotting function

    x_label: x_label as string

    y_label: y_label as string

    x_rot: x-tick rotation, default=None, can be int 0-360

    y_rot: y-tick rotation, default=None, can be int 0-360

    fontsize: size of plot font on axis, defaul=12, can be int/float

    fontweight: Adding character to font, default=None, can be 'bold'

    legend: Choice of including legend, default=None, bool, True:False

    save: Saves image output, default=False, bool

    save_name: Name of output image file as .png. Requires Save to be True.

               default=None, string: 'Insert Name.png'

    Output: A customized plot based on given parameters and an output file

    

    """

    #Ticks

    ax.tick_params(direction='out', length=5, width=3, colors='k',

               grid_color='k', grid_alpha=1,grid_linewidth=2)

    plt.xticks(fontsize=fontsize, fontweight=fontweight, rotation=x_rot)

    plt.yticks(fontsize=fontsize, fontweight=fontweight, rotation=y_rot)



    #Legend

    if legend==None:

        pass

    elif legend==True:

        plt.legend()

        ax.legend()

        pass

    else:

        ax.legend().remove()

        

    #Labels

    plt.xlabel(x_label, fontsize=fontsize, fontweight=fontweight, color='k')

    plt.ylabel(y_label, fontsize=fontsize, fontweight=fontweight, color='k')



    #Removing Spines and setting up remianing, preset prior to use.

    ax.spines['top'].set_color(None)

    ax.spines['right'].set_color(None)

    ax.spines['bottom'].set_color('k')

    ax.spines['bottom'].set_linewidth(3)

    ax.spines['left'].set_color('k')

    ax.spines['left'].set_linewidth(3)

    

    if save==True:

        plt.savefig(save_name)

    
fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block

plot=sns.scatterplot(data=iris, x=	'SepalLengthCm' ,y='SepalWidthCm', hue='Species', palette=['b','r','g'], s=30)#Scatter plot

Plotter(plot, 'SepalLength (cm)', 'SepalWidth (cm)', legend=True, save=True, save_name='Scatter.png')#Plotter function for aesthetics

plot
fig, ax=plt.subplots()

plot=sns.boxplot(data=iris, x=	'Species' ,y='SepalWidthCm', palette=sns.color_palette("mako"))#Generates box plot

plot=sns.swarmplot(data=iris, x=	'Species' ,y='SepalWidthCm', palette=['red'], marker='^')#Generates points as triangles

Plotter(plot, 'Species', 'SepalWidth (cm)',20,None,10,legend=None, save=True, save_name='Box-Swarm.png')

plot
fig, ax=plt.subplots()

plot=sns.histplot(iris, x='SepalLengthCm', hue='Species', element="step",stat="density",kde=True, common_norm=False,)

Plotter(plot, 'Sepal Length (cm)', 'Density',legend=None, save=True, save_name='histo.png')

plot
fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block

plot=sns.heatmap(iris.corr(),annot=True, cmap='Blues', linewidths=1)

Plotter(plot, None, None, 50,legend=False, save=True, save_name='Corr.png')

#plot=plt.hist(iris.groupby('Species')['SepalWidthCm'], label=iris['Species'])

#Plotter(plot, 'Species', 'SepalWidth (cm)',legend=True, save=True, save_name='Box-Swarm.png')

#plot
lter=pd.read_csv('/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_lter.csv')

lter.info()
print('***Species',lter['Species'].unique())

print('***Sex',lter['Sex'].unique())
new_lter=lter.copy()

#Species name

cleaner(new_lter,'Species',old_label1=['Adelie Penguin (Pygoscelis adeliae)'], new_label1='P.adeliae',old_label2=['Chinstrap penguin (Pygoscelis antarctica)'],

        new_label2='P.antartica',old_label3=['Gentoo penguin (Pygoscelis papua)'],new_label3='P.papua')

print(new_lter['Species'].unique())



#Culmen Length

cleaner(new_lter,'Culmen Length (mm)', group_by=True, category='Species', score='mean')#Fill in numerical data by species mean



#Culmen Depth

cleaner(new_lter,'Culmen Depth (mm)', group_by=True, category='Species', score='mean')



#Flipper Length

cleaner(new_lter,'Flipper Length (mm)', group_by=True, category='Species', score='mean')



#Body Mass

cleaner(new_lter,'Body Mass (g)', group_by=True, category='Species', score='mean')

new_lter['Body Mass (kg)']=new_lter['Body Mass (g)']/1000 #Conversion to kilograms



#Sex 

filler=['nan','.']

cleaner(new_lter,'Sex',old_label1=filler, new_label1=new_lter['Sex'].mode()[0],

        old_label2=['MALE'],new_label2='male', old_label3=['FEMALE'], new_label3='female')

print(new_lter['Sex'].unique())



#Delta 15N

cleaner(new_lter,'Delta 15 N (o/oo)', group_by=True, category='Species', score='mean')



#Delta 13C

cleaner(new_lter,'Delta 13 C (o/oo)', group_by=True, category='Species', score='mean')



#Dropping Features

new_lter=new_lter.drop(['Body Mass (g)','studyName','Sample Number','Comments','Individual ID',

                       'Region', 'Stage', 'Date Egg'], axis=1)



print(new_lter.info())#Lets check to make sure all NaN is gone

new_lter.head()#Visualize the table to confirm the changes we wanted
fig, ax=plt.subplots()

plot=sns.countplot(data=new_lter, x='Species', hue='Sex', palette=['darkblue','darkred'])

Plotter(plot, x_label='Species', y_label='Count',legend=False, save=True, save_name='Penguin Species Count.png')

plot
fig, ax=plt.subplots()

plot=sns.boxplot(data=new_lter, x="Species", y='Body Mass (kg)', palette=['darkblue','darkred','darkgreen'])

Plotter(plot,'Species', 'Body Mass (kg)',x_rot=20, legend=False, save=True, save_name='Species-Mass.png')

plot
#Label Encoding

from sklearn import preprocessing 

LE=preprocessing.LabelEncoder()



lter_encode=new_lter.copy()

lter_encode['Island']=LE.fit_transform(lter_encode['Island'])

lter_encode['Clutch Completion']=LE.fit_transform(lter_encode['Clutch Completion'])

lter_encode['Sex']=LE.fit_transform(lter_encode['Sex'])

lter_encode['Species_Code']=LE.fit_transform(lter_encode['Species']) #This will be used for a correlation matrix

lter_encode.head()
#Feature Selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

X=lter_encode.drop(['Species', 'Species_Code'], axis=1)

Y=lter_encode['Species']

bestfeatures = SelectKBest(score_func=f_classif, k='all')

fit = bestfeatures.fit(X,Y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']  #naming the dataframe columns

print(featureScores.nlargest(12,'Score'))  #print 10 best features



#Plotting Feature Importance

fig, ax = plt.subplots(figsize=(7,7))

plot=sns.barplot(data=featureScores, x='Score', y='Feature', palette='viridis',linewidth=0.5, saturation=2, orient='h')

Plotter(plot, 'Importance Score', 'Feature', legend=False, save=True, save_name='Feature Importance.png')

plot
fig, ax = plt.subplots(figsize=(7,7))

plot=sns.heatmap(lter_encode.corr(),annot=True, linewidths=0.1)

Plotter(plot,None,None,90, fontweight='bold',legend=None,save=True, save_name='Corr Matrix.png')

plot
OHE=preprocessing.OneHotEncoder()

lter_encode_drop=lter_encode.copy()

lter_encode_drop=lter_encode_drop.drop(['Sex','Clutch Completion','Species_Code'], axis=1)

lter_ohe=lter_encode_drop.copy()

lter_code=OHE.fit_transform(lter_ohe[['Island']]).toarray()

lter_list=list(sorted(new_lter['Island'].unique()))  #Will name the column values to the respective Island

lter_code=pd.DataFrame(lter_code, columns=lter_list)#Setting OHE dataframe for merge

lter_ohe=pd.concat([lter_code,lter_ohe], axis=1)#Merging Data Frames

lter_ohe=lter_ohe.drop(['Island'], axis=1)
#Splitting

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(lter_ohe.drop(['Species'], axis=1), lter_ohe['Species'],test_size=0.25, random_state=0)



#Scaling

scaler=preprocessing.StandardScaler()



X_train_scaled=scaler.fit_transform(X_train) #Scaling and fitting the training set to a model

X_test_scaled=scaler.transform(X_test) #Transformation of testing set based off of trained scaler model



#PCA Dimension Reduction

from sklearn.decomposition import PCA

n=5 #Number of components

pca=PCA(n_components=n)

X_train_pca=pca.fit_transform(X_train_scaled)# Fitting and transforming the training data

X_test_pca=pca.transform(X_test_scaled)# Transforming the test data by the fitted trained PCA()
from sklearn.svm import SVC #Classifier

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV #Paramterizers

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #Accuracy metrics

import itertools #Used for iterations
def Searcher(estimator, param_grid, search, train_x, train_y, test_x, test_y,label=None):

    """

    This is a helper function for tuning hyperparameters using the two search methods.

    Methods must be GridSearchCV or RandomizedSearchCV.

    Inputs:

        estimator: Any Classifier

        param_grid: Range of parameters to search

        search: Grid search or Randomized search

        train_x: input variable of your X_train variables 

        train_y: input variable of your y_train variables

        test_x: input variable of your X_test variables

        test_y: input variable of your y_test variables

        label: str to print estimator, default=None

    Output:

        Returns the estimator instance, clf

        

    Modified from: https://www.kaggle.com/crawford/hyperparameter-search-comparison-grid-vs-random#To-standardize-or-not-to-standardize

    

    """   

    from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV #Paramterizers

    from sklearn.metrics import accuracy_score, classification_report #Accuracy metrics

    import itertools #Used for iterations

    

    try:

        if search == "grid":

            clf = GridSearchCV(

                estimator=estimator, 

                param_grid=param_grid, 

                scoring=None,

                n_jobs=-1, 

                cv=10, #Cross-validation at 10 replicates

                verbose=0,

                return_train_score=True

            )

        elif search == "random":           

            clf = RandomizedSearchCV(

                estimator=estimator,

                param_distributions=param_grid,

                n_iter=10,

                n_jobs=-1,

                cv=10,

                verbose=0,

                random_state=1,

                return_train_score=True

            )

    except:

        print('Search argument has to be "grid" or "random"')

        sys.exit(0) #Exits program if not grid or random

        

    # Fit the model

    clf.fit(X=train_x, y=train_y)

    

    #Testing the model

    try:

        if search=='grid':

            cfmatrix=confusion_matrix(

            y_true=test_y, y_pred=clf.predict(test_x))

        

            #Defining prints for accuracy metrics of grid

            print("**Grid search results of", label,"**")

            print("The best parameters are:",clf.best_params_)

            print("Best training accuracy:\t", clf.best_score_)

            print('Classification Report:')

            print(classification_report(y_true=test_y, y_pred=clf.predict(test_x))

             )

        elif search == 'random':

            cfmatrix=confusion_matrix(

            y_true=test_y, y_pred=clf.predict(test_x))



            #Defining prints for accuracy metrics of grid

          

            print("**Random search results of", label,"**")

            print("The best parameters are:",clf.best_params_)

            print("Best training accuracy:\t", clf.best_score_)

            print('Classification Report:')

            print(classification_report(y_true=test_y, y_pred=clf.predict(test_x))

                 )

    except:

        print('Search argument has to be "grid" or "random"')

        sys.exit(0) #Exits program if not grid or random

        

    return clf, cfmatrix; #Returns a trained classifier with best parameters
def plot_confusion_matrix(cm, label,color=None,title=None):

    """

    Plot for Confusion Matrix:

    Inputs:

        cm: sklearn confusion_matrix function for y_true and y_pred as seen in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

        title: title of confusion matrix as a 'string', default=None

        label: the unique label that represents classes for prediction can be done as sorted(dataframe['labels'].unique()).

        color: confusion matrix color, default=None, set as a plt.cm.color, based on matplot lib color gradients

    """

    classes=sorted(label)

    plt.imshow(cm, interpolation='nearest', cmap=color)

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)

    plt.ylabel('Actual')

    plt.xlabel('Predicted')

    thresh = cm.mean()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j]), 

                 horizontalalignment="center",

                 color="white" if cm[i, j] < thresh else "black") 
#Grid Search SVM Parameters

svm_param = {

    "C": [.01, .1, 1, 5, 10, 100], #Specific parameters to be tested at all combinations

    "gamma": [0, .01, .1, 1, 5, 10, 100],

    "kernel": ["rbf","linear","poly"],

    "random_state": [1]}

svm_grid, cfmatrix_grid= Searcher(SVC(), svm_param, "grid", X_train_pca, y_train, X_test_pca, y_test,label='SVC')



print('_____'*20)

#Randomized Search SVM Parameters

svm_dist = {

    "C": np.arange(0.01,2, 0.01),   #By using np.arange it will select from randomized values

    "gamma": np.arange(0,1, 0.01),

    "kernel": ["rbf","linear","poly"],

    "random_state": [1]}

svm_rand, cfmatrix_rand= Searcher(SVC(), svm_dist, "random", X_train_pca, y_train, X_test_pca, y_test)



#Plotting the confusion matrices

plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix',label=lter_ohe['Species'].unique(), color=plt.cm.Greens) #grid matrix function

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=lter_ohe['Species'].unique(), color=plt.cm.Blues) #randomized matrix function



plt.savefig('confusion.png')