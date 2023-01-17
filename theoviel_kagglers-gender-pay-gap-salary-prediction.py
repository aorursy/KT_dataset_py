import warnings
import itertools
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style('whitegrid')
df_choice = pd.read_csv('../input/multipleChoiceResponses.csv')
df_choice.head()
print("Number of replies to the survey :", df_choice.shape[0])
question_names = df_choice.iloc[0]
df_choice = df_choice.drop(0, axis=0)
print(question_names['Q9'])
print(df_choice['Q9'].unique())
df_choice = df_choice[df_choice['Q9'].notnull()]
df_choice = df_choice[df_choice['Q9'] != 'I do not wish to disclose my approximate yearly compensation']
print(df_choice.shape[0], "replies left")
order = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000', 
  '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000', 
  '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000', 
  '300-400,000', '400-500,000', '500,000+']

plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q9'], order=order)
plt.xticks(rotation=-45)
plt.xlabel("Yearly Income ($)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Yearly income repartition", fontsize=15)
plt.show()
dic = {'0-10,000': 5000, '10-20,000': 15000, '20-30,000': 25000, '30-40,000': 35000, 
       '40-50,000': 45000, '50-60,000': 55000, '60-70,000': 65000, '70-80,000': 75000, 
       '80-90,000': 85000, '90-100,000': 95000, '100-125,000': 112500, 
       '125-150,000': 137500, '150-200,000': 175000, '200-250,000': 225000, 
       '250-300,000': 275000, '300-400,000': 350000, '400-500,000': 450000, 
       '500,000+':500000}

df_choice['target'] = df_choice['Q9'].apply(lambda x: dic[x])
liars = df_choice[df_choice['Q6'] == "Student"]
liars = liars[liars['target'] >= 500000]
liars.head(10)
df_choice = df_choice[df_choice['target'] < 500000]
print(question_names['Q1'])
plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q1'])
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Gender Repartition among the Kaggle Community", fontsize=15)
plt.show()
plt.show()
plt.figure(figsize=(15,10))
sns.violinplot(x='Q1', y='target', data=df_choice)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Gender", fontsize=12)
plt.title("Distribution of the Yearly income for Different Genders", fontsize=15)
plt.show()
print(question_names['Q2'])
order = ['18-21', '22-24', '25-29', '30-34','35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70-79', '80+']
plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q2'], order=order)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Age Repartition of Kagglers", fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
sns.boxplot(x='Q2', y='target', data=df_choice, order=order, showfliers=False)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Age", fontsize=12)
plt.title("Distribution of the Yearly Income for Age Groups", fontsize=15)
plt.show()
print(question_names['Q3'])
country_dic = {'Morocco': 'Africa',
             'Tunisia': 'Africa',
             'Austria': 'Europe',
             'Hong Kong (S.A.R.)': 'Asia',
             'Republic of Korea': 'Asia',
             'Thailand': 'Asia',
             'Czech Republic': 'Europe',
             'Philippines': 'Asia',
             'Romania': 'Europe',
             'Kenya': 'Africa',
             'Finland': 'Europe',
             'Norway': 'Europe',
             'Peru': 'South America',
             'Iran, Islamic Republic of...': 'Middle East',
             'Bangladesh': 'Asia',
             'New Zealand': 'Oceania',
             'Egypt': 'Africa',
             'Chile': 'South America',
             'Belarus': 'Europe',
             'Hungary': 'Europe',
             'Ireland': 'Europe',
             'Belgium': 'Europe',
             'Malaysia': 'Asia',
             'Denmark': 'Europe',
             'Greece': 'Europe',
             'Pakistan': 'Asia',
             'Viet Nam': 'Asia',
             'Argentina': 'South America',
             'Colombia': 'South America',
             'Indonesia': 'Oceania',
             'Portugal': 'Europe',
             'South Africa': 'Africa',
             'South Korea': 'Asia',
             'Switzerland': 'Europe',
             'Sweden': 'Europe',
             'Israel': 'Middle East',
             'Nigeria': 'Africa',
             'Singapore': 'Asia',
             'I do not wish to disclose my location': 'dna',
             'Mexico': 'North America',
             'Ukraine': 'Europe',
             'Netherlands': 'Europe',
             'Turkey': 'Asia',
             'Poland': 'Europe',
             'Australia': 'Oceania',
             'Italy': 'Europe',
             'Spain': 'Europe',
             'Japan': 'Asia',
             'France': 'Europe',
             'Canada': 'North America', 
             'United Kingdom of Great Britain and Northern Ireland': 'Europe',
             'Germany': 'Europe',
             'Brazil': 'South America',
             'Russia': 'Russia',
             'Other': 'Other',
             'China': 'China',
             'India': 'India',
             'United States of America': 'USA'}
df_choice['Q3'] = df_choice['Q3'].apply(lambda x: country_dic[x])
plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q3'], order=df_choice['Q3'].value_counts().index)
plt.xticks(rotation=-70)
plt.xlabel("Country / Region", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Where are Kagglers from ?", fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
sns.violinplot(x='Q3', y='target', data=df_choice, order=df_choice['Q3'].value_counts().index)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Nationality", fontsize=12)
plt.title("Distribution of the Yearly Income for Different Regions", fontsize=15)
plt.show()
df = df_choice[df_choice['Q1'] != "Prefer not to say"]
df = df[df['Q1'] != "Prefer to self-describe"]
plt.figure(figsize=(15,10))
sns.violinplot(x='Q3', y='target', hue='Q1', data=df, split=True, order=df_choice['Q3'].value_counts().index)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Nationality", fontsize=12)
plt.title("Illustration of the Gender Wage Gap for Different Regions", fontsize=15)
plt.show()
print(question_names['Q4'])
order = ['Doctoral degree', 'Master’s degree', 'Bachelor’s degree',  'Some college/university study without earning a bachelor’s degree',
         'Professional degree', 'No formal education past high school', 'I prefer not to answer']

plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q4'], order=order)
plt.xticks(rotation=-70)
plt.xlabel("Studies", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Which Level of Study do Kagglers Have ?", fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
sns.violinplot(x='Q4', y='target', data=df_choice, order=order)
plt.xticks(rotation=-70)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Nationality", fontsize=12)
plt.title("Distribution of the Yearly Income for Different Levels of Study", fontsize=15)
plt.show()
df = df[df['Q3'] == 'USA']

plt.figure(figsize=(15,10))
sns.violinplot(x='Q4', y='target', hue='Q1', data=df, split=True, order=order)
plt.xticks(rotation=-70)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Studies", fontsize=12)
plt.title("Illustration of the Gender Wage Gap for Different Levels of Education in the USA", fontsize=15)
plt.show()
print(question_names['Q5'])
plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q5'], order=df_choice['Q5'].value_counts().index)
plt.xticks(rotation=-80)
plt.xlabel("Major", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("What's Kagglers' Fields of Study ?", fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
sns.violinplot(x='Q5', y='target', data=df_choice, order=df_choice['Q5'].value_counts().index)
plt.xticks(rotation=-80)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Major", fontsize=12)
plt.title("Distribution of the Yearly Income for Different Fields of Study", fontsize=15)
plt.show()
print(question_names['Q6'])
plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q6'], order=df_choice['Q6'].value_counts().index)
plt.xticks(rotation=-70)
plt.xlabel("Profession", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("What's Kagglers' Job ?", fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
sns.boxplot(x='Q6', y='target', data=df_choice, order=df_choice['Q6'].value_counts().index, showfliers=False)
plt.xticks(rotation=-70)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Profession", fontsize=12)
plt.title("Distribution of the Yearly Income for Different Types of Jobs", fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
sns.violinplot(x='Q6', y='target', hue='Q1', data=df, split=True, order=df['Q6'].value_counts().index)
plt.xticks(rotation=-70)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Profession", fontsize=12)
plt.title("Illustration of the Gender Wage Gap for Different Professions in the USA", fontsize=15)
plt.show()
# Mean salary of each job
means = df.groupby(['Q6'])['target'].mean().sort_values(ascending=False)

# Women proportion of each job
d = {"Female":1, "Male":0}
df['Q1'] = df['Q1'].apply(lambda x: d[x])
women_perc = df.groupby(['Q6'])['Q1'].mean()

# Joining
df_job = pd.concat([means, women_perc], axis=1)
plt.figure(figsize=(15,10))
sns.barplot(x=df_job.index, y='Q1', data=df_job, order=means.index)
plt.xticks(rotation=-70)
plt.ylabel("Women proportion", fontsize=12)
plt.xlabel("Profession", fontsize=12)
plt.title("Percentage of Women in Jobs, Sorted by Average Salary in the USA ", fontsize=15)
plt.show()
# Linear regression
z = np.polyfit(df_job['Q1'], df_job['target'], 1)
p = np.poly1d(z)

plt.figure(figsize=(15,10))
plt.scatter(df_job['Q1'], df_job['target'], label='Samples')
plt.plot(np.arange(0, 0.6, 0.01), p(np.arange(0, 0.6, 0.01)), linestyle=':', label='Trend')
plt.ylabel("Average Yearly Income of the Job ($)", fontsize=12)
plt.xlabel("Percentage of Women in the Job", fontsize=12)
plt.title("In the USA, the Higher Earning the Job, the fewer Women", fontsize=15)
plt.legend()
plt.show()
print(question_names['Q7'])
plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q7'], order=df_choice['Q7'].value_counts().index)
plt.xticks(rotation=-70)
plt.xlabel("Industry of Employer", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Where do people work ?", fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
sns.boxplot(x='Q7', y='target', data=df_choice, order=df_choice['Q7'].value_counts().index, showfliers=False)
plt.xticks(rotation=-70)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Industry", fontsize=12)
plt.title("Distribution of the Yearly Income for Different Industries of Employment", fontsize=15)
plt.show()
question_names['Q8']
order = ['0-1', '1-2', '2-3',  '3-4', '4-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30 +']

plt.figure(figsize=(15,10))
sns.countplot(df_choice['Q8'], order=order)
plt.xlabel("Years of Experience", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("How Experienced are Kagglers in their Current Jobs ?", fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
sns.violinplot(x='Q8', y='target', data=df_choice, order=order)
plt.xticks(rotation=-70)
plt.ylabel("Yearly Income ($)", fontsize=12)
plt.xlabel("Profession", fontsize=12)
plt.title("Distribution of the Yearly Income in Function of the Years of Experience", fontsize=15)
plt.show()
features = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
target = ["target"]

df = df_choice[features + target]

df = df.fillna('?')
dic_age = {'30-34': 32, '22-24': 23, '35-39': 37, '18-21': 19.5, '40-44': 42, '25-29': 27, '55-59': 57, '60-69': 64.5, '45-49': 47, '50-54': 52, '70-79': 74.5, '80+': 80}
dic_exp = {'5-10': 7.5, '0-1': 0.5, '10-15': 12.5, '3-4': 3.5, '1-2': 1.5, '2-3': 2.5, '15-20': 17.5, '4-5': 4.5, '25-30': 27.5, '20-25': 22.5, '30 +': 30, '?': 0}

df['Q2'] = df['Q2'].apply(lambda x: dic_age[x])
df['Q8'] = df['Q8'].apply(lambda x: dic_exp[x])

for q in ["Q1", "Q3", "Q4", "Q5", "Q6", "Q7"]:
    df[q] = df[q].astype('category')
    
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
df = df.rename(index=str, columns={"Q1": 'Gender', "Q2": 'Age', "Q3": 'Country', "Q4": 'Education', "Q5": 'Major', "Q6": 'Profession', "Q7": 'Industry', "Q8": 'Experience'})
classes = ['less than 10k', 'between 10k and 30k', 'between 30k and 50k', 'between 50k and 80k', 'between 80k and 125k', 'more than 100k']
dic_target = {5000: 0,  
              15000: 1, 25000: 1, 
              35000: 2, 45000: 2, 
              55000: 3,  65000: 3,  75000: 3,
              85000: 4, 95000: 4, 112500: 4,
              137500: 5,  175000: 5, 225000: 5, 275000: 5, 350000: 5,  450000: 5
             }

df['target'] = df['target'].apply(lambda x: dic_target[x])
plt.figure(figsize=(15,10))
sns.countplot(df['target'])
plt.xticks(range(0, 7), classes)
plt.ylabel("Count", fontsize=12)
plt.xlabel("Yearly income ($)", fontsize=12)
plt.title("Reparition of our New Classes", fontsize=15)
plt.show()
df.head()
df_train, df_test = train_test_split(df, test_size=0.2)
print(f"Training on {df_train.shape[0]} samples.")
features = ['Gender', 'Age', 'Country', 'Education', 'Major', 'Profession', 'Industry', 'Experience']
      
def run_lgb(df_train, df_test):
    params = {"objective" : "multiclass",
              "num_class": 6,
              "metric" : "multi_error",
              "num_leaves" : 30,
              "min_child_weight" : 50,
              "learning_rate" : 0.05,
              "bagging_fraction" : 0.7,
              "feature_fraction" : 0.7,
              "bagging_seed" : 420,
              "verbosity" : -1
             }
    
    lg_train = lgb.Dataset(df_train[features], label=(df_train["target"].values))
    lg_test = lgb.Dataset(df_test[features], label=(df_test["target"].values))
    model = lgb.train(params, lg_train, 1000, valid_sets=[lg_test], early_stopping_rounds=100, verbose_eval=100)
    
    return model
model = run_lgb(df_train, df_test)
pred_train = model.predict(df_train[features], num_iteration=model.best_iteration)
pred_test = model.predict(df_test[features], num_iteration=model.best_iteration)
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.ylabel('Feature', size=12)
plt.xlabel('Importance', size=12)
plt.title("Importance of the Features our LightGBM Model", fontsize=15)
plt.show()
def plot_confusion_matrix(cm, classes, title='Confusion matrix', normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fmt = '.2f' if normalize else 'd'

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=15)
    plt.colorbar()
    plt.grid(False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = (cm.max()+cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)
conf_mat_train = confusion_matrix(np.argmax(pred_train, axis=1), df_train[target].values)

plot_confusion_matrix(conf_mat_train, classes, title='Confusion matrix on train data', normalize=True)
conf_mat_test = confusion_matrix(np.argmax(pred_test, axis=1), df_test[target].values)

plot_confusion_matrix(conf_mat_test, classes, title='Confusion matrix on test data', normalize=True)