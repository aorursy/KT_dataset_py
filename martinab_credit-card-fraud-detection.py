# Importing libraries:



# Importing numpy, pandas, matplotlib and seaborn:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Imports for plotly:

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots





# To keep graph within the nobebook:

%matplotlib inline



# To hide warnings

import warnings

warnings.filterwarnings('ignore')
# Read data from dataset's csv file:

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
# Display the shape of dataset (#of rows, #of columns):

print('The shape of the dataset is: ', df.shape)
# Show first 5 rows of dataset:

df.head()
# Explore dataframe's statistics (numerical values only):

df.describe()
# Function to describe variables

def desc(df):

    d = pd.DataFrame(df.dtypes,columns=['Data_Types'])

    d = d.reset_index()

    d['Columns'] = d['index']

    d = d[['Columns','Data_Types']]

    d['Missing'] = df.isnull().sum().values    

    d['Uniques'] = df.nunique().values

    return d



# Use desc() function to describe df:

desc(df)
# Class distribution dataframe:



cls_df = pd.DataFrame(df.groupby(['Class'])['Class'].count())

cls_df['Category'] = ['Not-Fraud', 'Fraud']



# Create bar chart for class distribution:



data=go.Bar( x = cls_df.Category

           , y = cls_df.Class

           ,  marker=dict( color=['#4c5cff', '#ff4682'])

           , text=cls_df.Class

           , textposition='auto' 

           )







layout = go.Layout( title = 'Class distribution'

                  , xaxis = dict(title = 'Class')

                  , yaxis = dict(title = 'Volume')

                  )



fig = go.Figure(data,layout)



fig.show()
# Count the occurrences of Fraud and Not-Fraud:

vol = df.Class.value_counts()



# Print the % of Not-Fraud and Fraud cases:

print(vol * 100 / len(df.index))
# Scatter graph for Distribution of Transactions split by Category (Not-Fraud/Fraud):



df['Category'] = df['Class'].map({0:'Not-Fraud', 1:'Fraud'})



fig = px.scatter(df

                 , x='V15'

                 , y='Amount'

                 , color = 'Category'

                 , size = 'Amount'

                 #, facet_col='Category'

                 , color_continuous_scale= ['#4c5cff','#ff4682']

                 , render_mode="webgl"

                )



fig.update_layout(title='Distribution of Transactions - by Category'

                  , xaxis_title='V15'

                  , yaxis_title='Transaction Value ($)'

                 )



fig.show()
# Correlation matrix for Credit Card - Fraud Detection dataset features:



corr = df.corr()

l = list(corr.columns)



fig = go.Figure(data=go.Heatmap(z=corr

                                , x=l

                                , y=l

                                , hoverongaps = False

                                

                               )

               )



fig.update_layout(title='Correlation for Features')





fig.show()
# Create dataset for Fraud & Non-Fraud features (mask_1&mask_0 respectively):

mask_1 = df.loc[df.Class == 1]

mask_0 = df.loc[df.Class == 0]



# Create a list of columns (excluding Class and Category):

columns = list(df.columns)[0:-2]
# Crate box plot and histograms for Fraud and Not-Fraud transactions:

for i in columns:

    fig = plt.figure(figsize = (12,4))



    plt.subplot(1,3,1)

    sns.boxplot(x='Class',

                y=i,

                hue='Category', 

                palette=  ['#ff4682','#4c5cff'],

                data=df

                )



    # Label axes

    plt.title('Boxplot '+'('+i+')')

    plt.xlabel('')

    plt.ylabel('value')

    

    plt.tight_layout() 



    plt.subplot(1,3,2)

    plt.hist(x=i

         , edgecolor = 'black'

         , linewidth = 1.5

         , bins = 20

         , color = '#ff4682'

         , label = 'Fraud'

        # , alpha = 0.5

         , data=mask_1)



    # Label axes

    plt.title('Fraud Histogram '+'('+i+')')

    plt.xlabel('')

    plt.ylabel('count')

    plt.tight_layout() 



    plt.subplot(1,3,3)

    plt.hist(x=i

         , edgecolor = 'black'

         , linewidth = 1.5

         , bins = 20

         , color = '#4c5cff'

         , label = 'Not-Fraud'

         , alpha = 0.5

         , data=mask_0)



    # Label axes

    plt.title('Non-Fraud Histogram '+'('+i+')')

    plt.xlabel('')

    plt.ylabel('count')

    

    plt.tight_layout()    

    



# Display label

plt.legend(loc='best')





plt.show()

X=df.drop(['Time', 'Class', 'Category'], axis = 1)

y=df.Class
# Import train_test_split form sklearn:

from sklearn.model_selection import train_test_split



# Keep 30% of data for testing:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)
# Count the occurrences of Fraud and Not-Fraud:

v = y_train.value_counts()



# Print the % of Not-Fraud and Fraud cases:

print(v * 100 / len(y_train.index))
from imblearn.over_sampling import SMOTE

smt = SMOTE()
# Transform the dataset



X, y = smt.fit_resample(X_train, y_train)
# Count the occurrences of Fraud and Not-Fraud:

v = y.value_counts()



# Print the % of Not-Fraud and Fraud cases:

print(v * 100 / len(y.index))
from sklearn.linear_model import LogisticRegression



# Importing classification_method and confusion_matrix:

from sklearn.metrics import classification_report, confusion_matrix
# Fit a logistic regression model to our data

model = LogisticRegression()

model.fit(X_train, y_train)



# Obtain model predictions

y_pred = model.predict(X_test)



# Print the classifcation report and confusion matrix

print('Classification report:\n', classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_true=y_test, y_pred = y_pred)

print('Confusion matrix:\n', conf_mat)
z = confusion_matrix(y_test, y_pred)



x = ['Genuine', 'Fraud']

y = ['Genuine', 'Fraud']



# change each element of z to type string for annotations

z_text = [[str(y) for y in x] for x in z]



# set up figure 

fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='RdBu')



# add title

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',

                  #xaxis = dict(title='x'),

                  #yaxis = dict(title='x')

                 )



# add custom xaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=0.5,

                        y=-0.15,

                        showarrow=False,

                        text="Predicted value",

                        xref="paper",

                        yref="paper"))





# add custom yaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=-0.35,

                        y=0.5,

                        showarrow=False,

                        text="Real value",

                        textangle=-90,

                        xref="paper",

                        yref="paper"))



# adjust margins to make room for yaxis title

fig.update_layout(margin=dict(t=50, l=200))



# add colorbar

fig['data'][0]['showscale'] = True

fig.show()
# Import the random forest model from sklearn

from sklearn.ensemble import RandomForestClassifier



# Define the model as the random forest

model = RandomForestClassifier(random_state=5)



# Fit the model to our training set

model.fit(X_train, y_train)



# Obtain predictions from the test data 

predicted = model.predict(X_test)

# Import the packages to get the different performance metrics

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score



# Predict probabilities

probs = model.predict_proba(X_test)



# Print the ROC curve, classification report and confusion matrix

print(roc_auc_score(y_test, probs[:,1]))

print(classification_report(y_test, predicted))

print(confusion_matrix(y_test, predicted))