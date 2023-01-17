# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import friedmanchisquare
from scipy import stats
from sklearn.metrics import confusion_matrix
 
 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
df = pd.read_csv('/kaggle/input/website-phishing-data-set/Website Phishing.csv')
# Replacing the phishy class by the value 2
# I decided to use MLP so it would be better if
# the classes were all non negative values

df['Result'] = df['Result'].astype(str).replace('-1','2').astype(np.int64)
# Visualizing the data
df.head()
# Visualizing the data
df.tail()
#  Calculating if there are null or na values in the dataset
print('Verifying null and na data')
print()
print(df.isna().any())

print()
print(df.isnull().any())

# Obtaining some additional information about the dataset
df.info()
# function to plot the class distribution by feature

def plot_class_distribution(feature, color, data, labels):

  class_info = data[feature].value_counts().sort_index()
  
  #x = class_info.index
  x = labels
  x_pos = [i for i, _ in enumerate(x)]

  y = class_info.values


  fig, ax = plt.subplots()
  rects1 = ax.bar(x_pos, y, color=color)
  # helper function to show the number of examples in each bar
  def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.f' % float(height),
        ha='center', va='bottom')
  autolabel(rects1)


  plt.ylabel("Number of Examples")
  plt.title(feature + " examples distribution\n")
  plt.xticks(x_pos, x)


legit = len(df[df['Result'] == 1])
susp = len(df[df['Result'] == 0])
phishy = len(df[df['Result'] == 2])

size=[legit, susp, phishy]
names = ['Legitimate', 'Suspicious', 'Phishy']
 
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(size, labels=names, colors=['blue','pink','red'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.rcParams["figure.figsize"] = (5,5)
plt.title('Class Distribution')
plt.show()

# Class distribution

legit = len(df[df['Result'] == 1])
susp = len(df[df['Result'] == 0])
phishy = len(df[df['Result'] == 2])

labels = 'Legitimate', 'Suspicious', 'Phishy'
sizes = [legit, susp, phishy]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
plt.title('Class Distribution')
plt.show()
# Class distribution

legit = len(df[df['Result'] == 1])
susp = len(df[df['Result'] == 0])
phishy = len(df[df['Result'] == 2])

x = ['Legitimate', 'Suspicious', 'Phishy']
x_pos = [i for i, _ in enumerate(x)]

y = [legit, susp, phishy]

fig, ax = plt.subplots()
rects1 = ax.bar(x_pos, y, color='lightblue')

plt.xlabel("Classes")
plt.ylabel("Number of Examples")
plt.title("Class distribution\n")
plt.xticks(x_pos, x)

def autolabel(rects):

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.f' % float(height),
        ha='center', va='bottom')
autolabel(rects1)

plt.show()
# Distribution of the column values according to the description from the paper


sfh_labels = ['Empty SFH', 'SFH different domain', 'Valid SFH']
plot_class_distribution('SFH', 'silver', df, sfh_labels)

pop_labels = ['Rightclick disabled', 'Rightclicl with alert', 'No pop-up']
plot_class_distribution('popUpWidnow', 'gold', df, pop_labels)

ssl_labels = ['Nor HTTP nor trusted', 'HTTP and nottrusted', 'HTTP and trusted']
plot_class_distribution('SSLfinal_State', 'silver', df, ssl_labels)

request_labels = ['req_URL > 61%',  '22 <= req_URL <= 61%', 'req_URL < 22%']
plot_class_distribution('Request_URL', 'gold', df, request_labels)

anchor_labels = [ 'Acr_URL>67%',' 31%<=Acr_URL<=67%', 'Acr_URL<31%']
plot_class_distribution('URL_of_Anchor', 'silver',df, anchor_labels)

web_labels = ['wtraffic>150K', 'wtraffic<=150K', 'wtraffic<150K']
plot_class_distribution('web_traffic', 'gold', df, web_labels)

url_labels = ['len > 75', '54 <= len <= 75', 'len < 54']
plot_class_distribution('URL_Length', 'silver', df, url_labels)

age_labels = ['age < 1 year', 'age > 1 year']
plot_class_distribution('age_of_domain', 'lightblue', df, age_labels)

ip_labels = ['No IPAdress URL','URL IPaddress']
plot_class_distribution('having_IP_Address', 'lightblue', df, ip_labels)

# Graph grouping the presence of IP address with the classes
 
dfip = df[df['having_IP_Address'] == 1]
dfnoip = df[df['having_IP_Address'] == 0]
 
labelsip = dfip['Result'].value_counts().index
valuesip = dfip['Result'].value_counts().values
 
labelsnoip = dfnoip['Result'].value_counts().index
valuesnoip = dfnoip['Result'].value_counts().values
 
 
barWidth = 0.25
 
bars1 = [ valuesip[0], valuesnoip[0]]
bars2 = [valuesip[1], valuesnoip[1]]
bars3 = [valuesip[2], valuesnoip[2]]
 
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#E4D9FF', width=barWidth, edgecolor='white', label='Phishy')
plt.bar(r2, bars2, color='lightblue', width=barWidth, edgecolor='white', label='Legitimate')
plt.bar(r3, bars3, color='silver', width=barWidth, edgecolor='white', label='Suspicious')
 
# Add xticks on the middle of the group bars
plt.xlabel('Class distribution by presence of IP addres in the URL', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))],['With IPaddress', 'Without IPaddress'] )
 
 
plt.legend()
plt.show()
# Graph with the relation between the Phishy class and the feature having IP address, and 
# URL length
dfphishy = df[df['Result'] == 2]

dfphishy_ip = dfphishy[dfphishy['having_IP_Address'] == 1]
dfphishy_noip = dfphishy[dfphishy['having_IP_Address'] == 0]


ip_values_url1_phis = list(dfphishy_ip['URL_Length'].value_counts().values)
noip_values_url1_phis = list(dfphishy_noip['URL_Length'].value_counts().values)

labels = ['URL < 54', '54 <= URL <= 75', 'URL > 75']
noip = noip_values_url1_phis
ip = ip_values_url1_phis

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ip, width, label='With IPaddress')
rects2 = ax.bar(x + width/2, noip, width, label='Without IPaddress')


ax.set_ylabel('Examples')
ax.set_title('Data from the Phishy class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

dflegitimate = df[df['Result'] == 1]

dflegit_ip = dflegitimate[dflegitimate['having_IP_Address'] == 1]
dflegit_noip = dflegitimate[dflegitimate['having_IP_Address'] == 0]


ip_values_url1_legit = list(dflegit_ip['URL_Length'].value_counts().values)
noip_values_url1_legit = list(dflegit_noip['URL_Length'].value_counts().values)
#ip_values_url1_legit.append(0)

labels = ['URL < 54', '54 <= URL <= 75', 'URL > 75']
noip = noip_values_url1_legit
ip = ip_values_url1_legit

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ip, width, label='With IPaddress')
rects2 = ax.bar(x + width/2, noip, width, label='Without IPaddress')


ax.set_ylabel('Examples')
ax.set_title('Data from Legitimate class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

dfsuspicious = df[df['Result'] == 0]

dfsusp_ip = dfsuspicious[dfsuspicious['having_IP_Address'] == 1]
dfsusp_noip = dfsuspicious[dfsuspicious['having_IP_Address'] == 0]


ip_values_url1_susp = list(dfsusp_ip['URL_Length'].value_counts().values)
noip_values_url1_susp = list(dfsusp_noip['URL_Length'].value_counts().values)
ip_values_url1_susp.append(0)

labels = ['URL < 54', '54 <= URL <= 75', 'URL > 75']
noip = noip_values_url1_susp
ip = ip_values_url1_susp

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ip, width, label='With IPaddress')
rects2 = ax.bar(x + width/2, noip, width, label='Without IPaddress')


ax.set_ylabel('Examples')
ax.set_title('Data from the Suspicious class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

######## Classification Functions ##########
 
 
# Function to load the classifiers
 
def initialize_classifiers():
  # KNN k = 3
  knn3 = KNeighborsClassifier(n_neighbors=3)
 
  # KNN k = 5
  knn5 = KNeighborsClassifier(n_neighbors=5)
 
  # KNN k = 7
  knn7 = KNeighborsClassifier(n_neighbors=7)
 
  # KNN ponderado k = 3
  knnp3 = KNeighborsClassifier(n_neighbors=3, weights='distance',metric='euclidean')
 
  
  # KNN ponderado k = 5
  knnp5 = KNeighborsClassifier(n_neighbors=5, weights='distance',metric='euclidean')
 
 
  # KNN ponderado k = 7
  knnp7 = KNeighborsClassifier(n_neighbors=7, weights='distance',metric='euclidean')
 
  # Support Vector Machine - função de kernel linear
  svmLinear = SVC(kernel='linear')
 
  # Support Vector Machine - função de kernel RBF
  svmRBF = SVC(kernel='rbf')
  
  # Decision Tree 
  decisionTree = DecisionTreeClassifier()
 
  # Random Forest
  randomForest = RandomForestClassifier()
 
  # Naïve Bayes
  naiveBayes = GaussianNB()
  
  # Logistic Regression
 
  logisticRegression = LogisticRegression()
 
  # MLP 
  modelo = Sequential()
  modelo.add(Dense(units=64, activation='relu',kernel_initializer='random_uniform', input_dim=9))
  modelo.add(Dense(units=32, activation='relu',kernel_initializer='random_uniform'))
  modelo.add(Dense(units=16, activation='relu',kernel_initializer='random_uniform'))
  modelo.add(Dense(units=3, activation='softmax'))
 
  optimizer = keras.optimizers.Adam(lr = 0.001, decay =0.0001, clipvalue= 0.5)
  modelo.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
 
 
  return [knn3, knn5, knn7,svmLinear, svmRBF, decisionTree, randomForest, naiveBayes, logisticRegression, modelo,knnp3,knnp5,knnp7]
 
# Function to train and store the evaluation results from the classifiers 
 
def train_and_evaluate(class_train, noclass_train, class_test, noclass_test, alg, dicio, epochs = 400, batch_size = 64, verb = False):
 
  if dicio['modelname'] == 'MLP':
    
    class_train_categorical = keras.utils.to_categorical(class_train, num_classes=3)
    class_test_categorical = keras.utils.to_categorical(class_test, num_classes=3)
    tempo_inicial = time.time()
    alg.fit(noclass_train, class_train_categorical,batch_size=batch_size, epochs=epochs, verbose= verb)
    tempo_fim = time.time()
    predicted = [np.argmax(pred) for pred in alg.predict(noclass_test)]
 
  else:
    tempo_inicial = time.time()
    alg.fit(noclass_train, class_train)
    tempo_fim = time.time()
 
    predicted = alg.predict(noclass_test)
 
  dicio['acc'].append(accuracy_score(class_test, predicted))
  dicio['fscore'].append(f1_score(class_test, predicted, average='macro'))
  dicio['precision'].append(precision_score(class_test, predicted, average='macro'))
  dicio['recall'].append(recall_score(class_test, predicted, average='macro'))
  dicio['tempo'].append(tempo_fim - tempo_inicial)
  dicio['cm'].append(confusion_matrix(class_test,predicted))
 
# Creating dictionaies to store the metric data to each one of the classifiers
 
def create_dictios():
 
  knn3_dc = {'modelname':'KNN3','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  knn5_dc = {'modelname':'KNN5','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  knn7_dc = {'modelname':'KNN7','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  knnp3_dc = {'modelname':'KNNP3','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  knnp5_dc = {'modelname':'KNNP5','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  knnp7_dc = {'modelname':'KNNP7','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  svmLinear_dc = {'modelname':'SVML','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  svmRBF_dc = {'modelname':'SVMR','acc': [], 'fscore': [], 'precision': [], 'recall': [] , 'tempo':[], 'cm': []}
  decisionTree_dc = {'modelname':'DT','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  randomForest_dc = {'modelname':'RF','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  naive_dc = {'modelname':'NB','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  logreg_dc = {'modelname':'LR','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
  mlp_dc = {'modelname':'MLP','acc': [], 'fscore': [], 'precision': [], 'recall': [], 'tempo':[], 'cm': []}
 
  return [knn3_dc, knn5_dc, knn7_dc,svmLinear_dc, svmRBF_dc, decisionTree_dc, randomForest_dc,  naive_dc, logreg_dc, mlp_dc, knnp3_dc,knnp5_dc,knnp7_dc]
 
 
######## Data visualization functions ##########
 
# Creating boxplot with the data from a metric
 
def create_boxplot(df, metric_name):
  df.boxplot(column= list(df.columns) , figsize=(12, 8))
  plt.xlabel('Algorithms')
  plt.ylabel(metric_name)
  plt.show()
 
  print()
  print()
  print()
 
# Creating the statistic test to a pair of classifiers
 
def create_statistic_test(measure, df):
 
  two_bests = list(df.mean().sort_values(ascending = False).index[:2])
  best = df[two_bests[0]].values
  secondbest = df[two_bests[1]].values
 
  pvalue = stats.ttest_ind(best, secondbest)
  if pvalue[1] < 0.05:
    print('The algorithms '+ two_bests[0] + ' and ' + two_bests[1] + ' are statistically different using the metric ' + measure)
    print('P-value: ' + str(pvalue[1]))
  else:
    print('The algorithms '+ two_bests[0] + ' and ' + two_bests[1] + ' are  NOT statistically different using the metric '  + measure)
    print('P-value: ' + str(pvalue[1]))
 
# Plotting the mean, median and standard deviation to each one of the classifiers used
 
def plot_stats_values(df, plot_name):
  media = df.mean()
  mediana = df.median()
  desviopadrao = df.std()
 
  barWidth = 0.25
 
  bars1 = media.values
  bars2 = mediana.values
  bars3 = desviopadrao.values
 
  labels = media.index
  
  # Set position of bar on X axis
  r1 = np.arange(len(bars1))
  r2 = [x + barWidth for x in r1]
  r3 = [x + barWidth for x in r2]
  
  # Make the plot
  plt.bar(r1, bars1, color='#E4D9FF', width=barWidth, edgecolor='white', label='Mean')
  plt.bar(r2, bars2, color='lightblue', width=barWidth, edgecolor='white', label='Median')
  plt.bar(r3, bars3, color='silver', width=barWidth, edgecolor='white', label='Std')
  
  # Add xticks on the middle of the group bars
  plt.xlabel(plot_name, fontweight='bold')
  plt.xticks([r + barWidth for r in range(len(bars1))],labels )
  
  plt.rcParams["figure.figsize"] = (20,3)
  plt.legend()
  plt.show()
 
# Plotting the metrics of accuracy, precision, fscore, recall and execution time to
# the three best classifiers

def plot_best_algs_result(df_acc, df_precision, df_fscore, df_recall, df_tempo):
  
  # searching fot the best classifiers using the accuracy mean
  bests = list(df_acc.mean().sort_values(ascending = False).index[:3])
  df_mean = df_acc.mean()
  media_precision = df_precision.mean()
  media_fscore = df_fscore.mean()
  media_recall = df_recall.mean()
  
  best = df_mean[bests[0]]
  secondbest = df_mean[bests[1]]
  thirdbest = df_mean[bests[2]]
 
 
  bests_acc = []
  bests_acc.append(best)
  bests_acc.append(secondbest)
  bests_acc.append(thirdbest)
 
  bests_precision = []
  bests_precision.append(media_precision[bests[0]])
  bests_precision.append(media_precision[bests[1]])
  bests_precision.append(media_precision[bests[2]])
 
  bests_recall = []
  bests_recall.append(media_recall[bests[0]])
  bests_recall.append(media_recall[bests[1]])
  bests_recall.append(media_recall[bests[2]])
 
  bests_fscore = []
  bests_fscore.append(media_fscore[bests[0]])
  bests_fscore.append(media_fscore[bests[1]])
  bests_fscore.append(media_fscore[bests[2]])
  
 
  barWidth = 0.15
 
  bars1 = bests_acc
  bars2 = bests_precision
  bars3 = bests_fscore
  bars4 = bests_recall
 
  labels = bests
 
  # Set position of bar on X axis
  r1 = np.arange(len(bars1))
  r2 = [x + barWidth for x in r1]
  r3 = [x + barWidth for x in r2]
  r4 = [x + barWidth for x in r3]
 
 
  # Make the plot
  plt.bar(r1, bars1, color='#E4D9FF', width=barWidth, edgecolor='white', label='Accuracy')
  plt.bar(r2, bars2, color='lightblue', width=barWidth, edgecolor='white', label='Precision')
  plt.bar(r3, bars3, color='silver', width=barWidth, edgecolor='white', label='F1-score')
  plt.bar(r4, bars4, color='gold', width=barWidth, edgecolor='white', label='Recall')
 
 
  # Add xticks on the middle of the group bars
  plt.xlabel(xlabel='Best Algorithms', fontweight='bold')
  plt.xticks([r + barWidth for r in range(len(bars1))],labels )
 
 
 
  plt.rcParams["figure.figsize"] = (20,10)
  plt.legend()
  plt.show()
# Splitting the data into the features and classes
df_noclass = df.iloc[:, 0:9]

df_class = df.iloc[:, 9]

# Classification with k-fold cross validation

# Initializing the classifiers
listmodels = initialize_classifiers()

# Creating the dictionaries that will store the classifiers' results 
listofdicts = create_dictios()

# Initializing the K-fold
kfold = StratifiedKFold(10, True, 1)

c = kfold.split(df_noclass, df_class)

for train_index, test_index in c:

  noclass_train, noclass_test =np.array(df_noclass.iloc[train_index]) ,np.array(df_noclass.iloc[test_index])
  class_train, class_test = np.array(df_class.iloc[train_index]), np.array(df_class.iloc[test_index])

  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[0], listofdicts[0])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[1], listofdicts[1])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[2], listofdicts[2])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[3], listofdicts[3])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[4], listofdicts[4])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[5], listofdicts[5])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[6], listofdicts[6])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[7], listofdicts[7])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[8], listofdicts[8])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[9] , listofdicts[9], epochs = 400, batch_size = 64, verb = False)
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[10], listofdicts[10])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[11] , listofdicts[11])
  train_and_evaluate(class_train, noclass_train, class_test, noclass_test, listmodels[12] , listofdicts[12])

# creating the dataframes that will store the evaluation metrics results

df_acc = pd.DataFrame(np.array([dic['acc'] for dic in listofdicts]).T, columns= [dic['modelname'] for dic in listofdicts])

df_fscore = pd.DataFrame(np.array([dic['fscore'] for dic in listofdicts]).T, columns=[dic['modelname'] for dic in listofdicts])

df_recall = pd.DataFrame(np.array([dic['recall'] for dic in listofdicts]).T, columns=[dic['modelname'] for dic in listofdicts])

df_precision = pd.DataFrame(np.array([dic['precision'] for dic in listofdicts]).T, columns=[dic['modelname'] for dic in listofdicts])

df_time = pd.DataFrame(np.array([dic['tempo'] for dic in listofdicts]).T, columns=[dic['modelname'] for dic in listofdicts])

# Making the statistic test to each one of the evaluation metrics
create_statistic_test('f1-score', df_fscore)
print()

create_statistic_test('recall', df_recall)
print()

create_statistic_test('precision', df_precision)
print()

create_statistic_test('accuracy', df_acc)

# Graphs with the mean, median, and standart deviation to each metric used to all the 
# tested classifirers

plot_stats_values(df_time, 'Time')

plot_stats_values(df_acc, 'Accuracy')

plot_stats_values(df_fscore, 'F1-score')

plot_stats_values(df_precision, 'Precision')

plot_stats_values(df_recall, 'Recall')


# Boxplot with the algorithms results to each one of the metrics used
create_boxplot(df_acc, metric_name='Accuracy')

create_boxplot(df_fscore, metric_name='F1-score')

create_boxplot(df_precision, metric_name='Precision')

create_boxplot(df_recall, metric_name='Recall')

plot_best_algs_result(df_acc, df_precision, df_fscore, df_recall, df_time)
# Mean accuracy of the best algorithms
media = df_acc.mean()

mlp = media['MLP']
rf = media['RF'] 
dt = media['DT']

x = ['MLP', 'Random Forest', 'Decision Tree']
x_pos = [i for i, _ in enumerate(x)]

y = [mlp, rf, dt]

fig, ax = plt.subplots()
rects1 = ax.bar(x_pos, y, color='pink')

plt.xlabel("Algorithms")
plt.ylabel("Accuracy mean")
plt.title("Best Algorithms\n")
plt.xticks(x_pos, x)

def autolabel(rects):

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.00*height,
                '%.3f' % float(height),
        ha='center', va='bottom')
autolabel(rects1)

plt.show()
# confusion matrix to the MLP algorithm (best algorithm)

cm = listofdicts[9]['cm'][3]
sns.heatmap(cm,annot=True)
# Using the K-means algorithm to cluster the data
kmeans3 = KMeans(n_clusters = 3)

# Using Principal Component Analysis to plot a 2D graph of the data
pca = PCA(n_components=2).fit(df_noclass)
pca_2d = pca.transform(df_noclass)

array_classe = np.array(df_class)

y_km = kmeans3.fit_predict(df_noclass)
# Plotting the result of the K-means 

plt.scatter(
    pca_2d[y_km == 0, 0], pca_2d[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='o', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    pca_2d[y_km == 1, 0], pca_2d[y_km == 1, 1],
    s=50, c='gold',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    pca_2d[y_km == 2, 0], pca_2d[y_km == 2, 1],
    s=50, c='red',
    marker='o', edgecolor='black',
    label='cluster 3'
)

plt.scatter(
    kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1],
    s=250, marker='*',
    c='black', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
# Plotando a distribuição de classes original
plt.scatter(
    pca_2d[array_classe == 0, 0], pca_2d[array_classe == 0, 1],
    s=50, c='lightgreen',
    marker='o', edgecolor='black',
    label='Suspicious'
)

plt.scatter(
    pca_2d[array_classe == 1, 0], pca_2d[array_classe == 1, 1],
    s=50, c='gold',
    marker='o', edgecolor='black',
    label='Legitimate'
)

plt.scatter(
    pca_2d[array_classe == 2, 0], pca_2d[array_classe == 2, 1],
    s=50, c='red',
    marker='o', edgecolor='black',
    label='Phishy'
)

plt.legend(scatterpoints=1)
plt.grid()
plt.show()