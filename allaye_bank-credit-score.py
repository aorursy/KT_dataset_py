import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error,r2_score, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
training = pd.read_csv('../input/my-dataset/credit_train.csv')

train_columns = training.columns
for columns in enumerate(train_columns):
    print(columns)

training.head()
training.tail(10)
print('The shape of the dataset is {} and the size is {}'.format(training.shape, training.size) )
vb = training.isna().sum()
vb
training.info()
training.describe()
training.dtypes
def calculate_null_values(dataframe):
    d_frame = dataframe
    # get the sum of the null value of  each column 
    d_frame_null_values = pd.DataFrame(dataframe.isna().sum())
    # reset the dataframe index
    d_frame_null_values.reset_index(inplace=True)
    # add colume header to the dataframe
    d_frame_null_values.columns = ['Field_names', 'Null_value']
    #calculate the percentage of null or missing values 
    d_frame_null_value_percentage = dataframe.isnull().sum() / len(dataframe) * 100
    d_frame_null_value_percentage = pd.DataFrame(d_frame_null_value_percentage)
    d_frame_null_value_percentage.reset_index(inplace=True)
    d_frame_null_value_percentage = pd.DataFrame(d_frame_null_value_percentage)
    d_frame_null_values['Null_values_percentage'] = d_frame_null_value_percentage[0]
#d_frame_null_values['neww']= d_frame_null_values['Null_value'].apply(lambda d_frame_null_values:(d_frame_null_values/len(training))*100)
    return d_frame_null_values
    
    
calculate_null_values(training)
training[training['Credit Score']> 850]
def credit_error(value):
    credit_value = value
    credit_value['Credit Score'] = np.where(value['Credit Score'] > 850, value['Credit Score'].values /10, value['Credit Score'])
    return credit_value
    
    
c_training = credit_error(training)
c_training.describe()
training.drop(columns=['Months since last delinquent','Loan ID','Customer ID'],axis=1, inplace=True )
training.head(10)

calculate_null_values(training)
training.tail(516)
training.drop(training.tail(514).index, inplace=True)
training.tail(516)
calculate_null_values(training)
training.describe()
training.interpolate(inplace=True)
calculate_null_values(training)
training['Years in current job'].hist(figsize=(10,10))
training['Years in current job'].describe()
training['Years in current job'].fillna('10+ years', inplace=True)
calculate_null_values(training)
sbn.pairplot(training)

sbn.countplot(x='Home Ownership',data=training)
training['Purpose'].value_counts().sort_values(ascending=True).plot(kind='barh', 
                    title="Purpose for Loans", figsize=(15,10))
training['Years in current job']=training['Years in current job'].str.extract(r"(\d+)")
training['Years in current job'] =training['Years in current job'].astype(float)
training
sbn.heatmap(training.corr())
sbn.distplot(training['Years of Credit History'])
training
#train = training
#train.drop(['Term', 'Home Ownership', 'Purpose'], axis=1, inplace=True)
cat_data = ['Loan Status','Term','Home Ownership','Purpose']
transformer = ColumnTransformer([('transform', OneHotEncoder(), cat_data )],  remainder = 'passthrough')
tra=  np.array(transformer.fit_transform(training), dtype = np.float)

tra = pd.DataFrame(tra)

tra
training
training['Loan Status'].hist()
training['Term'].hist()
training['Home Ownership'].hist()
rename={1:'Paid',0:'Charged Off',2:'Long Term',3:'Short Term',5:'Home Mortgage',6:'Own Home', 7:'Rent',4:'Have Mortage'
        ,13:'Home Improvements', 11:'Debt Consolidation',19:'other', 15:'Other', 17:'major_purchase', 21:'small_business'
        ,14:'Medical Bills', 8:'Business Loan', 9:'Buy House', 10:'Buy a Car', 16:'Take a Trip', 23:'wedding', 22:'vacation'
        ,18:'moving', 12:'Educational Expenses', 20:'renewable energy', 24:"Loan Amount", 25:'credit score'
        ,26:'Annaual InCOME',27:'Years in Job',28:'monthly debt',29:'credit history',30:'Open account',31:'credit Problem'
        ,32:'Current credit balance',33:'Maximum open credit',34:'Bankruptcies',35:'Tax Liens', }

tra.rename(columns=rename, inplace=True)
tra
#pd.set_option('display.max_rows',)
#aaaa= tra[[12,20]]
#aaaa =tra[[]]
#asaa= np.array(trr)
#asaa[:]
#aaaa.head(98565)
tra.drop(columns=['Charged Off', 'Long Term', 'Have Mortage', 'renewable energy'], axis=1, inplace=True)
dependent = tra['Paid']
feature = tra.drop(columns=['Paid'])
x_train, x_test, y_train, y_test = train_test_split(feature,dependent, test_size=0.25, random_state=0)
print('The x_train shape is {} and the x_test shape is {} while the y_train shape is {} and the y_test shape is {}'
     .format(x_train.shape,x_test.shape,y_train.shape,x_test.shape))



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train.to_numpy()
y_test.to_numpy()
y_train
x_train
x_test
y_train
y_test
# Takes in a classifier, trains the classifier, and evaluates the classifier on the dataset
def do_prediction(classifier):
    
    # training the classifier on the dataset
    classifier.fit(x_train, y_train)
    
    #Do prediction and evaluting the prediction
    prediction = classifier.predict(x_test)
    evaluate_prediction = cross_val(x_train,y_train, classifier)
    coff_metrix = confusion_matrix(y_test, prediction)
    
    return evaluate_prediction,coff_metrix

def cross_val(x_train, y_train, classifier):
    # Applying k-Fold Cross Validation
    
    accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 5)
    return accuracies.mean()

#Training and Making prediction with Logistic Regression classifier
logreg = LogisticRegression(random_state=0)
evaluate_logreg, log_metric = do_prediction(logreg)
print('LogisticRegression Performace on the test_set have an accuracy score of {}'.format((evaluate_logreg *100).round()) )
group_names = ["True Neg","False Pos","False Neg",'True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                log_metric.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     log_metric.flatten()/np.sum(log_metric)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = plt.axes()
sbn.heatmap(log_metric, annot=labels, fmt='', cmap='Blues', ax=ax)
ax.set_title('Logistic Regression Confussion Metrix')
knn= KNeighborsClassifier(n_neighbors=7, p=2, metric='minkowski')
evaluate_knn, knn_metric = do_prediction(knn)
print('KNeighborsClassifier Performace on the test_set have an accuracy score of {}'.format((evaluate_knn *100).round()) )
group_names = ["True Neg","False Pos","False Neg",'True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                knn_metric.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     knn_metric.flatten()/np.sum(knn_metric)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = plt.axes()
sbn.heatmap(knn_metric, annot=labels, fmt='', cmap='Blues', ax=ax)
ax.set_title('KNeighbors Confussion Metrix')
gaussian = GaussianNB()
evaluate_gaussian, gaussian_metric = do_prediction(gaussian)
print('GaussianNB Performace on the test_set have an accuracy score of {}'.format((evaluate_gaussian *100).round()) )
group_names = ["True Neg","False Pos","False Neg",'True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                gaussian_metric.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     gaussian_metric.flatten()/np.sum(gaussian_metric)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = plt.axes()
sbn.heatmap(gaussian_metric, annot=labels, fmt='', cmap='gist_heat_r', ax=ax)
ax.set_title('GaussianNB Confussion Metrix')

rand = RandomForestClassifier()
evaluate_rand, rand_metric= do_prediction(rand)
print('RandomForest Performace on the test_set have an accuracy score of {}'.format((evaluate_rand *100).round()) )
group_names = ["True Neg","False Pos","False Neg",'True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                rand_metric.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     rand_metric.flatten()/np.sum(rand_metric)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = plt.axes()
sbn.heatmap(rand_metric, annot=labels, fmt='', cmap='twilight', ax=ax)
ax.set_title('RandomForest Confussion Metrix')
gboost = GradientBoostingClassifier()
evaluate_gboost, gboost_metric = do_prediction(gboost)
print('GradientBoosting performace on the test_set have an accuracy score of {}'.format((evaluate_gboost *100).round()) )
group_names = ["True Neg","False Pos","False Neg",'True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                gboost_metric.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     gboost_metric.flatten()/np.sum(gboost_metric)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = plt.axes()
sbn.heatmap(gboost_metric, annot=labels, fmt='', cmap='icefire', ax=ax)
ax.set_title('GradientBoost Confussion Metrix')
d_tree = DecisionTreeClassifier()
d_tree.maxi_dept=100
evaluate_d_tree, d_tree_metric = do_prediction(d_tree)
print('DecisiomTree performace on the test_set have an accuracy score of {}'.format((evaluate_d_tree *100).round()) )
group_names = ["True Neg","False Pos","False Neg",'True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                d_tree_metric.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     d_tree_metric.flatten()/np.sum(d_tree_metric)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = plt.axes()
sbn.heatmap(d_tree_metric, annot=labels, fmt='', cmap='terrain', ax=ax)
ax.set_title('DecisionTree Confussion Metrix')
xboost = XGBClassifier()
evaluate_xboost, xboost_metric = do_prediction(xboost)
print('XBoost Classifier performace on the test_set have an accuracy score of {}'.format((evaluate_xboost *100).round()) )
group_names = ["True Neg","False Pos","False Neg",'True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                xboost_metric.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     xboost_metric.flatten()/np.sum(xboost_metric)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = plt.axes()
sbn.heatmap(xboost_metric, annot=labels, fmt='', cmap='binary', ax=ax)
ax.set_title('XBoost Classifier Confussion Metrix')
plt.style.use('fivethirtyeight')
figsize=(8, 6)

# Dataframe to hold the results
weigh_up = pd.DataFrame({'model': ['Logistic Regression', 'K-NN', 'Decision Tree','Gradiant Boost', 'Random Forest',
                                  'GaussianNG'],
                        'score': [evaluate_logreg, evaluate_knn, evaluate_d_tree
                                  ,evaluate_gboost,evaluate_rand,evaluate_gaussian]})

# Horizontal bar chart of test mae
weigh_up.sort_values('score', ascending = True).plot(x = 'model', y = 'score', kind = 'barh',
                                                           color = 'red', edgecolor = 'black')

# Plot formatting
plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('K-Fold Cross Validation'); plt.xticks(size = 14)
plt.title('Model Comparison on Score', size = 20);
