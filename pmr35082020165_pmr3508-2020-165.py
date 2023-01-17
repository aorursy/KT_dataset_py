# Importing libs



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

plt.style.use('seaborn')
# Importing data



train_data = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = '?')

train_data.set_index('Id',inplace=True)

train_data
# Checking missing data



total = train_data.isna().sum().sort_values(ascending = False)

percent = ((train_data.isna().sum()/train_data.isna().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
# Checking distribution of missing data columns



print('occupation:\n')

print(train_data['occupation'].describe())



print('\n\nworkclass:\n')

print(train_data['workclass'].describe())



print('\n\nnative.country:\n')

print(train_data['native.country'].describe())
# Filling missing values with mode due to it's high frequency



value = train_data['workclass'].mode()[0]

train_data['workclass'] = train_data['workclass'].fillna(value)



value = train_data['native.country'].mode()[0]

train_data['native.country'] = train_data['native.country'].fillna(value)



value = train_data['occupation'].mode()[0]

train_data['occupation'] = train_data['occupation'].fillna(value)
# Checking missing data again



total = train_data.isna().sum().sort_values(ascending = False)

percent = ((train_data.isna().sum()/train_data.isna().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

sns.set()

sns.pairplot(train_data, vars = cols, hue = 'income', diag_kws={'bw':'1.5'})
# Describing the data



train_data.describe()
# Due to capital.gain's and capital.loss's mean and std, we notice that they have distant and well defined groups.

# So we'll check their histograms.
# Checking capital.gain histogram



plt.figure(figsize=(13, 7))

train_data['capital.gain'].hist(color = 'coral')

plt.xlabel('capital gain')

plt.ylabel('quantity')

plt.title('Capital gain histogram')
# Checking capital.loss histogram



plt.figure(figsize=(13, 7))

train_data['capital.loss'].hist(color = 'coral')

plt.xlabel('capital loss')

plt.ylabel('quantity')

plt.title('Capital loss histogram')
# As we can see, there's an almost absolute concentration on small values, while there's a few on high values
# Checking age curve with .hist()



plt.figure(figsize=(13, 7))

train_data['age'].hist(color = 'coral')

plt.xlabel('age')

plt.ylabel('quantity')

plt.title('Age histogram')
# Checking age curve with .distplot() (seaborn) and compraing with a distribution curve



sns.set()

plt.figure(figsize=(13,7))

sns.distplot(train_data['age'], color='darkorchid', bins = 70)

plt.ylabel('quantity')

plt.title('Distribution of age')
# We notice that there is a large quantity of people aged from 20 to 40 years old.
# Checking hours per week histogram



plt.figure(figsize=(13, 7))

train_data['hours.per.week'].hist(color = 'coral')

plt.xlabel('hours per week')

plt.ylabel('quantity')

plt.title('Hours per week histogram')
# We also notice that most of the subjects from the data works about 40 hours per week, which is healthy.

# Although, there is a considerable amount of subjects that goes over this number
# Checking subjects that works more than 40 hours per week



over_work = train_data[train_data['hours.per.week'] > 40]

plt.figure(figsize=(13, 7))

over_work['hours.per.week'].hist(color = 'coral', bins = 5)

plt.xlabel('hours per week')

plt.ylabel('quantity')

plt.title('Hours per week histogram')
# Calculating over workers average working hours per week



mean = over_work['hours.per.week'].mean()

mean
# As we can see, these over workers work an average of 53 hours per week.

# Even though this group of workers drops exponentially with hours of work, working more than 40 hours per week starts to get exhausting.
# Now we'll compare the data between the subjects using histograms, pie charts and boxplots.
# Creating comparison function for histograms



def compare_histogram(df, obj_var, test_var, obj_labels = None, alpha = 0.7):

    

    if obj_labels is None:

        obj_labels = df[obj_var].unique()

    

    #obj_var = 'income'

    #obj_labels = ['>50K', '<=50K']

    #test_var = 'age' (for example)

    

    temp = []

    n_labels = len(obj_labels)

    for i in range(n_labels):

        temp.append(df[df[obj_var] == obj_labels[i]])

        temp[i] = np.array(temp[i][test_var]).reshape(-1,1)



    fig = plt.figure(figsize= (13,7))

    

    for i in range(n_labels):

        plt.hist(temp[i], alpha = alpha)

    plt.xlabel(test_var)

    plt.ylabel('quantity')

    plt.title('Histogram over \'' + test_var + '\' filtered by \'' + obj_var + '\'')

    plt.legend(obj_labels)
# Comparing income's and age's histograms



compare_histogram(train_data, 'income', 'age')
# We can notice with this histogram, that the distribution of people with more than 50K of income is a normal distribution with the expectation on the age of 45.

# While the distribution of people with income less or equal than 50K decreases with increasing age
# Checking the sex distribution



plt.figure(figsize=(13, 7))

train_data['sex'].value_counts().plot(kind = 'pie')
# Calculating male and female quantities



male = train_data[train_data['sex'] == 'Male'].count()[0]



female = train_data.shape[0] - male



print('male qty: ',male)

print('female qty: ',female)
# As we can see, theres a lot more males than females subjects.
#Comparing sex histograms with income



compare_histogram(train_data, 'income', 'sex')
# Calculating the percentage of males with salaries over 50K and of women with salaries over 50K, and comparing it.



# male_income = [qty > 50K, qty <= 50K]

male_income = []

temp = train_data[train_data['sex'] == 'Male']

male_income.append(temp[temp['income'] == '>50K'].count()[0])

male_income.append(male-male_income[0])



# female_income = [qty > 50K, qty <= 50K]

female_income = []

temp = train_data[train_data['sex'] == 'Female']

female_income.append(temp[temp['income'] == '>50K'].count()[0])

female_income.append(female-female_income[0])



# % of male that has >50K income:

male_over = male_income[0]/male



# % of female that has >50K income:

female_over = female_income[0]/female



print(male_over*100,'% of male that has >50K income')

print(female_over*100,'% of female that has >50K income')
#Comparing sex histograms with working hours per week



compare_histogram(train_data, 'sex', 'hours.per.week')
# We note that there are a lot more men, in quantity and in percentage, that works more than 40 hours per week

# And, in percentage, more women working less than 30 hours per week
female = train_data[train_data['sex'] == 'Female']

male = train_data[train_data['sex'] == 'Male']
# Analysing occupations between genders



plt.figure(figsize=(13, 7))

male['occupation'].value_counts().plot(kind = 'bar', color = 'purple')

plt.ylabel('quantity')

plt.title('Histogram of male over occupations')



plt.figure(figsize=(13, 7))

female['occupation'].value_counts().plot(kind = 'bar', color = 'coral')

plt.ylabel('quantity')

plt.title('Histogram of female over occupations')
# We can see that there's a clear constrast of employability between the genders.
# Checking ethnic data



plt.figure(figsize=(13, 7))

train_data['race'].value_counts().plot(kind = 'pie')
# We notice a clear predominance on white race among the subjects
# Comparing ethnic histograms with income



compare_histogram(train_data, 'income', 'race')
# Calculating the percentage of subjects of each ethnicity with salaries over 50K



white = train_data[train_data['race'] == 'White'].count()[0]

black = train_data[train_data['race'] == 'Black'].count()[0]

amer = train_data[train_data['race'] == 'Amer-Indian-Eskimo'].count()[0]

other = train_data[train_data['race'] == 'Other'].count()[0]

asian = train_data[train_data['race'] == 'Asian-Pac-Islander'].count()[0]



# kind_income = [qtd > 50K, qtd <= 50K]

white_income = []

temp = train_data[train_data['race'] == 'White']

white_income.append(temp[temp['income'] == '>50K'].count()[0])

white_income.append(white-white_income[0])



black_income = []

temp = train_data[train_data['race'] == 'Black']

black_income.append(temp[temp['income'] == '>50K'].count()[0])

black_income.append(black-black_income[0])



amer_income = []

temp = train_data[train_data['race'] == 'Amer-Indian-Eskimo']

amer_income.append(temp[temp['income'] == '>50K'].count()[0])

amer_income.append(amer-amer_income[0])



asian_income = []

temp = train_data[train_data['race'] == 'Asian-Pac-Islander']

asian_income.append(temp[temp['income'] == '>50K'].count()[0])

asian_income.append(asian-asian_income[0])



other_income = []

temp = train_data[train_data['race'] == 'Other']

other_income.append(temp[temp['income'] == '>50K'].count()[0])

other_income.append(other-other_income[0])



print('White %: ', white_income[0]*100/white)

print('Black %: ', black_income[0]*100/black)

print('Amer-Indian-Eskimo %: ', amer_income[0]*100/amer)

print('Asian-Pac-Islander %: ', asian_income[0]*100/asian)

print('Other %: ', other_income[0]*100/other)
# As we can see, there's a certain social inequality, where Black, Amer-Indian-Eskimo , Others has proportionally lower salaries than the other ethnicities.
white = train_data[train_data['race'] == 'White']

black = train_data[train_data['race'] == 'Black']
# Comparing the occupations of the White and The Black ethnicities



plt.figure(figsize=(13, 7))

white['occupation'].value_counts().plot(kind = 'bar', color = 'purple')

plt.ylabel('quantity')

plt.title('Histogram of white people over occupations')



plt.figure(figsize=(13, 7))

black['occupation'].value_counts().plot(kind = 'bar', color = 'coral')

plt.ylabel('quantity')

plt.title('Histogram of black people over occupations')
#Again we notice a large difference between the occupations of each ethnicity.
#Comparing the education of the both ethnicities



var1 = 'race'

var2 = 'education.num'



data = pd.concat([train_data[var2], train_data[var1]], axis=1)



f, ax = plt.subplots(figsize=(15, 7))



sns.boxplot(x=var1, y=var2, data=data, notch = True)

plt.title('Boxplot of education num over race')
# Again whites and Asian-Pac-Islander have a higher mean than the others ethnicities
over_50k = train_data[train_data['income'] == '>50K']

under_50k = train_data[train_data['income'] == '<=50K']
# Checking the occupations's salaries



plt.figure(figsize=(13, 7))

over_50k['occupation'].value_counts().plot(kind = 'bar', color = 'purple')

plt.ylabel('quantity')

plt.title('Histogram of income over 50K over occupations')



plt.figure(figsize=(13, 7))

under_50k['occupation'].value_counts().plot(kind = 'bar', color = 'coral')

plt.ylabel('quantity')

plt.title('Histogram of income under 50K over occupations')
# As we can see, the occupations that have an income of more than 50K mostly are Exec-managerial and Prof-specialty, which are predominantly occupied by white males.

# While the the occupations that have an income of lower than 50K mostly are Adm-clerical, Craft-repair, Other-service and Sales, which are predominantly occupied by women and blacks.
# Analysing the ages with working hours per week



var2 = 'age'

var1 = 'hours.per.week'



data = pd.concat([train_data[var2], train_data[var1]], axis=1)



f, ax = plt.subplots(figsize=(14, 15))



sns.boxplot(x=var1, y=var2, data=data, orient = 'h')

plt.title('Boxplot of age over hours per week')
# We can notice that young and old people works on average less than 40 hours per week, probably beacuse they work half period.
# Analysing education level with working hours per week



var2 = 'education'

var1 = 'hours.per.week'



data = pd.concat([train_data[var2], train_data[var1]], axis=1)



f, ax = plt.subplots(figsize=(13, 7))

#ax.set_ylim(0,10000)



order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',

         '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',

         'Some-college', 'Bachelors', 'Masters', 'Doctorate']

sns.boxplot(x=var1, y=var2, data=data, order = order)
# We observe that those with higher education are most likely to work more than 40 hours per week.
# We'll use libs as skitlearn in this section to preprocess some data and to analyse some correlations between attributes, to optimaze the machine learning. 
# Let's preprocess some attributes



# education:

# Qualitative and ordered

# Preschool < 1st-4th < 5th-6th < 7th-8th < 9th < 10th < 11th < 12th < HS-grad < Prof-school < Assoc-acdm < Assoc-voc < Some-college < Bachelors < Masters < Doctorate



# workclass:

# Boolean

# isPrivate



# age

# Quantitative and ordered

# Young (0-25) < Middle-aged (26-45) < Senior (46-65) < Old (66+)



# hours.per.week

# Quantitative and ordered

# Part-time (0-25) < Full-time (25-40) < Over-time (40-60) < Too-much (60+)



# capital.gain and capital.loss

# Quantitative and ordered

# None (0) < Low (0 - mediana dos valores maiores que zero) and High (> mediana dos valores maiores que zero)
train_data['income'].value_counts()
base = train_data
base.columns
# Separating qualitative and quantitative attributes



quantitative_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

qualitative_columns = ['education', 'marital.status', 'occupation', 'relationship', 'race','sex', 'native.country', 'income']
# Declaring some functions to process the data



def isprivate(value):

    if value == 'Private':

        return 1

    return 0



def catg(value, categories, ordenation = None):

    if ordenation is not None:

        ordenation = np.arange(0, len(categories))

    for pos in ordenation:

        if value == categories[pos]:

            return pos

    return -1



def equals(value, x):

    for v in x:

        if v == value:

            return 1

    return 0
base['workclass'].unique()
# private: 1 if works for private sector, 0 otherwise

private = pd.DataFrame({'private': base['workclass'].apply(isprivate)})
train_data['native.country'].value_counts()
# countries: 1 if it's global_south, 0 otherwise

countries = ['Mexico', 'Philippines', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'Jamaica',

             'South', 'China', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Taiwan',

             'Haiti', 'Iran', 'Nicaragua', 'Peru', 'Ecuador', 'Trinadad&Tobago', 'Cambodia',

             'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras']

global_south = pd.DataFrame({'sul.global': base['native.country'].apply(equals, args = [countries])})
base['education'].unique()
# education



education_order = [15, 11, 5, 12, 10, 1, 14, 7, 2, 8, 4, 13, 0, 3, 6, 9]

args = [base['education'].unique(), education_order]

education_classes = pd.DataFrame({'education.classes': base['education'].apply(catg, args = args)})
# hours.per.week



aux = pd.cut(base['hours.per.week'], bins = [-1, 25, 40, 60, 200], labels = [0, 1, 2, 3])

hours_per_week_clusters = pd.DataFrame({'hours.per.week.clusters': aux})

hours_per_week_clusters = hours_per_week_clusters.astype(np.int)
# capital.gain and capital.loss



median = np.median(base[base['capital.gain'] > 0]['capital.gain'])

aux = pd.cut(base['capital.gain'],

             bins = [-1, 0, median, base['capital.gain'].max()+1],

             labels = [0, 1, 2])

capital_gain_clusters = pd.DataFrame({'capital.gain.clusters': aux})

capital_gain_clusters = capital_gain_clusters.astype(np.int)



median = np.median(base[base['capital.loss'] > 0]['capital.loss'])

aux = pd.cut(base['capital.loss'],

             bins = [-1, 0, median, base['capital.loss'].max()+1],

             labels = [0, 1, 2])

capital_loss_clusters = pd.DataFrame({'capital.loss.clusters': aux})

capital_loss_clusters = capital_loss_clusters.astype(np.int)
new_data = pd.concat([global_south, private, education_classes, 

                      hours_per_week_clusters, capital_gain_clusters, 

                      capital_loss_clusters], axis = 1)

new_data.head()
aux = base['income'].apply(equals, args = [['>50K']])



aux = pd.concat([new_data, pd.DataFrame({'income': aux})], axis = 1)



new = aux.astype(np.int)

aux.head()
# Visualizing a correlation matrix



corr_mat = aux.corr()

corr_mat

sns.set()

plt.figure(figsize=(10,8))

sns.heatmap(corr_mat, annot=True)
# Merging base with constructed values and selection of some attributes by intuition



base = base.drop(['fnlwgt', 'education', 'sex', 'native.country', 'workclass', 'marital.status'], axis = 1)

base = pd.concat([new_data, base], axis = 1)

base.head()
# Classification data encoding using LabelEnconder()



from sklearn import preprocessing as prep



names = ['occupation', 'relationship', 'race']

enc_x = []

for i in range(len(names)):

    enc_x.append(prep.LabelEncoder())

enc_y = prep.LabelEncoder()



i = 0

for name in names:

    base[name] = enc_x[i].fit_transform(base[name])

    i += 1



base['income'] = enc_y.fit_transform(base['income'])



base.head()
# Correlation matrix between the attributes



aux = base.astype(np.int)



corr_mat = aux.corr()

f, ax = plt.subplots(figsize=(20, 13))

sns.heatmap(corr_mat, vmax=.7, square=True, cmap="coolwarm", annot = True)
# Removing not relevant attributes



unselected_columns = []

unselected_columns.append('capital.loss')

unselected_columns.append('capital.gain')

unselected_columns.append('sul.global')

unselected_columns.append('private')

unselected_columns.append('education.classes')

unselected_columns.append('hours.per.week.clusters')



base = base.drop(unselected_columns, axis = 1)

base.head()
aux = base.astype(np.int)



corr_mat = aux.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_mat, vmax=.7, square=True, cmap="coolwarm")
# We'll use the method k-NN from sklearn for the machine learning

# To evaluate performance we will use cross validation
# We'll preprocess the data for StandardScaler use



from sklearn.preprocessing import StandardScaler



base.shape
X = base.drop(['income'], axis = 1)

y = base['income']



scaler_x = StandardScaler()



X = scaler_x.fit_transform(X)
# We'll evaluate the learning for k from 1 to 29, to identify the best k value for predicting the test base.

# The evaluation will be executed by using the cross validation with 5 folds, this way will figure out the best hyperparameter for k.
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
scores_mean = []

scores_std = []



k_lim_inf = 1

k_lim_sup = 30



folds = 5



k_max = None

max_std = 0

max_acc = 0



i = 0

print('Finding best k...')

for k in range(k_lim_inf, k_lim_sup):

    

    KNNclf = KNeighborsClassifier(n_neighbors=k, p = 1)

    

    score = cross_val_score(KNNclf, X, y, cv = folds)

    

    scores_mean.append(score.mean())

    scores_std.append(score.std())

    

    if scores_mean[i] > max_acc:

        k_max = k

        max_acc = scores_mean[i]

        max_std = scores_std[i]

    i += 1

    if not (k%3):

        print('   K = {0} | Best CV acc = {1:2.2f}% +/-{3:4.2f}% (best k = {2})'.format(k, max_acc*100, k_max, max_std*100))

print('\nBest k: {}'.format(k_max))
# Ploting a graphic with the average errors from the cross validation for different k values to find out the best one



plt.figure(figsize=(15, 7))

plt.errorbar(np.arange(k_lim_inf, k_lim_sup), scores_mean, scores_std,

             marker = 'o', markerfacecolor = 'purple' , linewidth = 3,

             markersize = 10, color = 'coral', ecolor = 'purple', elinewidth = 1.5)





yg = []

x = np.arange(0, k_lim_sup+1)

for i in range(len(x)):

    yg.append(max_acc)

plt.plot(x, yg, '--', color = 'purple', linewidth = 1)

plt.xlabel('k')

plt.ylabel('accuracy')

plt.title('KNN performed on several values of k')

plt.axis([0, k_lim_sup, min(scores_mean) - max(scores_std), max(scores_mean) + 1.5*max(scores_std)])
# Training the k-NN classifier with our train data and infering it´s accuracy



k = k_max



KNNclf = KNeighborsClassifier(n_neighbors=k, p = 1)

KNNclf.fit(X, y)
# We'll predict the class values from the test database with our trained classifier
# Importing data



test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values='?')

test_data.set_index('Id', inplace = True)

test_data.head()
# Adding extra columns



# capital.gain.cluster

median = np.median(test_data[test_data['capital.gain'] > 0]['capital.gain'])

aux = pd.cut(test_data['capital.gain'],

             bins = [-1, 0, median, test_data['capital.gain'].max()+1],

             labels = [0, 1, 2])

capital_gain_clusters = pd.DataFrame({'capital.gain.clusters': aux})

capital_gain_clusters = capital_gain_clusters.astype(np.int)



# capital.loss.cluster

median = np.median(test_data[test_data['capital.loss'] > 0]['capital.loss'])

aux = pd.cut(test_data['capital.loss'],

             bins = [-1, 0, median, test_data['capital.loss'].max()+1],

             labels = [0, 1, 2])

capital_loss_clusters = pd.DataFrame({'capital.loss.clusters': aux})

capital_loss_clusters = capital_loss_clusters.astype(np.int)



new_data = pd.concat([capital_gain_clusters, capital_loss_clusters], axis = 1)
features = ['age', 'education.num', 'occupation', 'relationship', 'race', 'hours.per.week']



base_test = pd.concat([new_data, test_data[features]], axis = 1)
base_test.head()
# Cheking missing data



total = base_test.isnull().sum().sort_values(ascending = False)

percent = ((base_test.isnull().sum()/base_test.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
# Filling missing values with mode



value = base_test['occupation'].mode()[0]

base_test['occupation'] = base_test['occupation'].fillna(value)
total = base_test.isnull().sum().sort_values(ascending = False)

percent = ((base_test.isnull().sum()/base_test.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

missing_data.head()
# Encoding



names = ['occupation', 'relationship', 'race']



i = 0

for name in names:

    base_test[name] = enc_x[i].transform(base_test[name])

    i += 1
base_test.head()
# Predicting the classes



X_prev = scaler_x.transform(base_test.values)
temp = KNNclf.predict(X_prev)



temp = enc_y.inverse_transform(temp)

temp = {'Income': temp}

predictions = pd.DataFrame(temp)
predictions.head()
# Submiting the result



predictions.to_csv("submission.csv", index = True, index_label = 'Id')