# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

us_gdp = pd.read_csv('../input/dwdm-week2-visualizations/US GDP.csv')



# Get a summary of numerical variables.

us_gdp.describe()



#preview the first 8 records of the dataset

us_gdp.head(8)

#preview the first 8 records of the dataset

us_gdp.tail(8)



us_gdp.iloc[33:21] # gives the data for a specific row.



us_gdp.loc[33:21]



us_gdp.plot(kind="scatter", x='Year', y='US_GDP_BN')



us_gdp.Year

us_gdp['Year']

print("")



us_gdp.T # transpose data values





us_gdp.plot(kind="scatter", # or `us_gdp.plot.scatter(`

x='Year', y='US_GDP_BN',

title="US GDP per year",

figsize=(10,8) )

plt.title("From %d to %d" % (

us_gdp['Year'].min(),

us_gdp['Year'].max()),size=8) 

plt.suptitle("US GDP per Year",size=12)

plt.ylabel("GDP")



us_gdp.Year.astype('str') # convert values into string data type



us_gdp.Year.astype('float') # convert values into float data type





us_gdp.Year.min() # show the minimum of the dataset



us_gdp.Year.min() # show the maximum of the dataset



axes = us_gdp.plot(kind='line', x='Year', y='US_GDP_BN')

us_gdp.plot(kind='scatter', x='Year', y='US_GDP_BN', ax=axes)

plt.title("From %d to %d" % (us_gdp['Year'].min(), us_gdp['Year'].max()))

plt.ylabel('US GDP per year')



us_gdp.plot(kind="line", # or `us_gdp.plot.line(`

x='Year', y='US_GDP_BN', title="US GDP per year", figsize=(10,8))

plt.title("From %d to %d" % (us_gdp['Year'].min(), us_gdp['Year'].max()

),size=8)

plt.suptitle("US GDP per Year",size=12)

plt.ylabel("GDP")



# The use of box plots to understand the distributions

us_gdp.boxplot(column='GDP_Growth_PC')



us_gdp['GDP_Growth_PC'].hist(bins=10)                                     



# Looking at GDP_Growth_PC and Year correlation.

us_gdp.boxplot(column='GDP_Growth_PC', by = 'Year')



# The use of histogram and boxplot of GDP_Growth_PC.

us_gdp['GDP_Growth_PC'].hist(bins=10)





# ------------------------------------ 5 Feb 2019 -------------------------------------------------#

# Segment dataset

gdp_group_pc_binned =pd.cut(us_gdp['GDP_Growth_PC'], 5)

gdp_group_pc_binned.value_counts()



us_gdp_year_groups = pd.cut(us_gdp['Year'], 3,

                            labels=['Early 1960s', 'Late 1900s to Earky 2000','After Early 2000s'])



print("=" *32)

print('Distribution per group')

print("=" *32)



pd.DataFrame(us_gdp_year_groups.value_counts())



us_gdp_with_year_groups_data = pd.DataFrame({'Time_Period':us_gdp_year_groups,

                                             'US_GDP_BN':us_gdp['US_GDP_BN']})

# Preview the first 5 records of the dataset us_gdp_with_year_groups_data

us_gdp_with_year_groups_data.head(5)



us_gdp_with_year_groups_data.boxplot(by='Time_Period')

plt.title('US GDP distribution in each time period')

plt.suptitle('')

plt.annotate('This is the smallest', (1, 6000))

print('')



# Group data by Time_Period

us_gdp_with_year_groups_data.groupby('Time_Period').describe()

















# Data for customizinig pie chart

labels = 'R Introduction', 'R Visualization', 'Python Introduction', 'Python Visualization'

sizes = [225, 130, 245, 210]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.1, 0, 0.1, 0)  # explode 1st slice

explode_gdp_group = (0.1, 0.1, 0.2, 0.1, 0.2)  # explode 1st slice

 

# Pie plot explode example

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()



gdp_group_pc_binned.value_counts().plot.pie(autopct='%2.1f%%', explode=explode_gdp_group)

print('\n\n')



                           
gdp_group_pc_binned.value_counts().plot.bar(align='center', alpha=0.5, title='GDP_Growth_PC')
# ----------------- Week 3 Decision Tree Python --------------------------------------------#

# --------------------------- Feb 12, 2018 -------------------------------------------------#

data_path = '../input/dwdm-week-3/Creditcardprom.csv' # Path to data file

data = pd.read_csv(data_path)  # read data from an excel file



data #view the data

# What columns are in the data set ? Do they have spaces that I should consider

data.columns
# The first two(2) rows have invalid data. Let us perform casewise deletion to remove these rows from the data set

data = data.drop([0,1],axis=0) # dropping items 0 and 1 from axis 0 or the x axis (rows)

data #viewing data
# We are only interested in a few columns

# extracting only sex, age and income,range, watch promo and life insurance promo

data2 = data[['Income Range','Sex','Age', 'Watch Promo', 'Life Ins Promo']]

data2



def create_label_encoder_dict(df):

    # Use to encode categorical data with value between 0 and n_classes-1.

    from sklearn.preprocessing import LabelEncoder

    

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
label_encoders = create_label_encoder_dict(data2)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)

    
# Apply each encoder to the data set to obtain transformed values

data3 = data2.copy() # create copy of initial data set

for column in data3.columns:

    if column in label_encoders:

        data3[column] = label_encoders[column].transform(data3[column])



print("Transformed data set")

print("="*45)

data3



# separate our data into dependent (Y) and independent(X) variables

X_data = data3[['Income Range','Sex','Age', 'Watch Promo']]

Y_data = data3['Life Ins Promo']



# Show Y and X data axises

X_data

Y_data

# We will split the data using a 70/30 split. i.e. 70% of the data will be

# randomly chosen to train the model and 30% will be used to evaluate the model



from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)

X_train.shape
y_train.shape
# import Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
# Create the classifier with a maximum depth of 2 using entropy as the criterion for choosing most significant nodes

# to build the tree

# Ensure the model is not overfitted "min_samples_split"

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=2) 

# Hint : Change the max_depth to 10 or another number to see how this affects the tree
# Build the classifier  by training it on the training data

clf.fit(X_train, y_train)
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100) ], 

    index = X_data.columns, columns = ['Feature Significance in Decision Tree'])
import graphviz
dot_data = tree.export_graphviz(clf,out_file=None, 

                                feature_names=X_data.columns, 

                         class_names=label_encoders[Y_data.name].classes_,  

                         filled=True, rounded=True,  proportion=True,

                                node_ids=True, #impurity=False,

                         special_characters=True)
graph = graphviz.Source(dot_data) 

graph
print("-" *50)

print('The above decision tree output suggested that people of lower income range [0 to 1] would be interesting in : ', '[Life Ins Promo]')
data2['Sex'].unique()
X_data.columns
gon = {

    'Income Range':'40-50,000', 

    'Sex':'M', 'Age':51, 

    'Watch Promo':'No'

}

gon

# Need to complete



gon_transformed = {

    'Income Range':label_encoders['Income Range'].transform([gon['Income Range']])[0], 

    'Sex':0,

    'Age':40,

    'Watch Promo':label_encoders['Watch Promo'].transform([gon['Watch Promo']])[0],

    

    }

gon_transformed
clf.predict(pd.DataFrame([gon_transformed]))
gon['Income Range']
# Building a Confusion Matrix

# NB. Data should be split in training and test data.

#The model built should be evaluated using unseen or test data



k=(clf.predict(X_test) == y_test) # Determine how many were predicted correctly
k.value_counts()



from sklearn.metrics import confusion_matrix


cm=confusion_matrix(y_test, clf.predict(X_test), labels=y_test.unique())

cm
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    import itertools

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
plot_confusion_matrix(cm,data2['Life Ins Promo'].unique())
