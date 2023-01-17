import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import math
perf = pd.read_csv('../input/StudentsPerformance.csv')
perf.info()
perf.head()
# A Function that will help us in making visualizations
def visualizer(x, y, plot_type, title, xlabel, ylabel, rotation=False, rotation_value=60, figsize=(15,8)):
    plt.figure(figsize=figsize)
    
    if plot_type == "bar":  
        sns.barplot(x=x, y=y)
    elif plot_type == "count":  
        sns.countplot(x)
    elif plot_type == "reg":  
        sns.regplot(x=x,y=y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    if rotation == True:
        plt.xticks(fontsize=13,rotation=rotation_value)
    plt.show()


corr_perf = perf.corr()

plt.figure(figsize=(15,5))
sns.heatmap(corr_perf)
corr_perf['writing score']
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
sns.distplot(perf['reading score'])

plt.subplot(1,2,2)
sns.distplot(perf['writing score'])
def ecdf(arr):
    arr = np.array(arr)
    n = len(arr)
    x = np.sort(arr)
    y = np.arange(1,n+1)/n
    return x, y

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
read_x, read_y = ecdf(perf["reading score"])
plt.plot(read_x, read_y, linestyle="none", color="red", marker="D", alpha=0.3)
plt.title("ECDF of Reading Score")

plt.subplot(1,2,2)
write_x, write_y = ecdf(perf["writing score"])
plt.plot(write_x, write_y, linestyle="none", color="blue", marker="D", alpha=0.3)
plt.title("ECDF of Writing Score")
def permutation_sample(data1, data2):

    data = np.concatenate((data1, data2))
    permuted_data = np.random.permutation(data)
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

reading_perm, writing_perm = permutation_sample(np.array(perf['reading score']),
                                                np.array(perf['writing score']))

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

read_x, read_y = ecdf(perf["reading score"])
plt.plot(read_x, read_y, linestyle="none", color="red", marker="D", alpha=0.3)

perm_read_x, perm_read_y = ecdf(reading_perm)
plt.plot(perm_read_x, perm_read_y, linestyle="none", color="green", marker="D", alpha=0.3)

plt.title("ECDF of Reading Score VS. Pemutation sample of reading score")

plt.subplot(1,2,2)
write_x, write_y = ecdf(perf["writing score"])
plt.plot(write_x, write_y, linestyle="none", color="blue", marker="D", alpha=0.3)

perm_write_x, perm_write_y = ecdf(writing_perm)
plt.plot(perm_write_x, perm_write_y, linestyle="none", color="green", marker="D", alpha=0.3)

plt.title("ECDF of Writing Score VS. Permutation sample of writing score")
visualizer(x=perf['race/ethnicity'],y=None, plot_type='count', 
           title="Ethicity distribution of students", xlabel="Group", ylabel="Count")
perf["race/ethnicity"].value_counts()
visualizer(x=perf['gender'],y=None, plot_type='count', 
           title="Gender distribution of students", xlabel="Gender", ylabel="Count")
perf["gender"].value_counts()
visualizer(x=perf['lunch'],y=None, plot_type='count', 
           title="Distribution of type of Lunch", xlabel="Lunch", ylabel="Count")
perf["lunch"].value_counts()
visualizer(x=perf['parental level of education'],y=None, plot_type='count', 
           title="Distribution of Parental level of education", xlabel="Parental Education", ylabel="Count")
perf["parental level of education"].value_counts()
visualizer(x=perf["test preparation course"],y=None, plot_type='count', 
           title="Distribution of Test Preparation Course completed", xlabel="Course", ylabel="Count")
perf["test preparation course"].value_counts()
plt.figure(figsize=(20,8))

plt.subplot(1,3,1)
sns.violinplot(y=['math score'], data=perf)
plt.title("Math Score",fontsize=18)

plt.subplot(1,3,2)
sns.violinplot(y=['reading score'], data=perf, color="green")
plt.title("Reading Score",fontsize=18)

plt.subplot(1,3,3)
sns.violinplot(y=['writing score'], data=perf, color="red")
plt.title("Writing Score",fontsize=18)
# Function that prints summary statistics of column given in parameters
def summary_statistics(col, df):
    print('Mean: {}'.format(df[col].mean()))
    print('Max: {}'.format(df[col].max()))
    print('Min: {}'.format(df[col].min()))
    print('Median: {}'.format(df[col].median()))
    print()
    
    # Number of students with maximum and minimum score
    max_score = df[col].max()
    min_score = df[col].min()
    print("Number of students who scored maximum score: {}".format(df[col][df[col]==max_score].count()))
    print("Number of students who scored minimum score: {}".format(df[col][df[col]==min_score].count()))
    print()
    
    # Students close to mean i.e. Students that have scores equal to floor(mean score) or ceiling(mean score)
    near_mean_floor = math.floor(df[col].mean())
    near_mean_ceil = math.ceil(df[col].mean())
    near_mean_tot = df[col][df[col]==near_mean_floor].count() + df[col][df[col]==near_mean_ceil].count()
    print("Number of students close to mean score: {}".format(near_mean_tot))
    print()
    
    # Students that have 50th percentile
    print("Number of students at median score: {}".format(df[col][df[col]==df[col].median()].count()))
    
    # Students with 25th percentile and 75th percentile scores
    print("Number of students at 25th percentile: {}".format(df[col][df[col]==df[col].quantile(0.25)].count()))
    print("Number of students at 75th percentile: {}".format(df[col][df[col]==df[col].quantile(0.75)].count()))

summary_statistics("math score", perf)
summary_statistics("reading score", perf)
summary_statistics("writing score", perf)
#Students that have more than median marks in 

# Maths
print("Maths")
perf_top_math = perf[perf["math score"] > perf["math score"].median()]
print(perf_top_math["test preparation course"].value_counts())
print()

# Reading
print("Reading")
perf_top_read = perf[perf["reading score"] > perf["reading score"].median()]
print(perf_top_read["test preparation course"].value_counts())
print()

# Writing
print("Writing")
perf_top_writ = perf[perf["writing score"] > perf["writing score"].median()]
print(perf_top_writ["test preparation course"].value_counts())
print()
#Students that have less than or equal to median marks in  

# Maths
print("Maths")
perf_bot_math = perf[perf["math score"] <= perf["math score"].median()]
print(perf_bot_math["test preparation course"].value_counts())
print()

# Reading
print("Reading")
perf_bot_read = perf[perf["reading score"] <= perf["reading score"].median()]
print(perf_bot_read["test preparation course"].value_counts())
print()

# Writing
print("Writing")
perf_bot_writ = perf[perf["writing score"] <= perf["writing score"].median()]
print(perf_bot_writ["test preparation course"].value_counts())
print()
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
#saving original dataset
perf_original = perf.copy()
perf['complete score'] = perf['math score'] + perf['reading score'] + perf['writing score']
perf.head()
def new_cat(data):
    if data in ["master's degree", "bachelor's degree", "some college"] :
        return "college"
    else :
        return "school"
    
perf["parental education"] = perf["parental level of education"].apply(new_cat)   
perf.head()
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin): #From Hands on Machine Learning
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

class group_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self):
        return self
    def transform(self):
        pass
    def fit_transform(self, X, y=None):
        self.encoder_par_edu = LabelEncoder()
        par_edu_encoded = self.encoder_par_edu.fit_transform(X['parental level of education'])
        
        self.encoder_test_prep = LabelEncoder()
        test_prep_encoded = self.encoder_test_prep.fit_transform(X['test preparation course'])
        
        self.encoder_lunch = LabelEncoder()
        lunch_encoded = self.encoder_lunch.fit_transform(X['lunch'])
        
        self.encoder_ethn = LabelEncoder()
        ethn_encoded = self.encoder_ethn.fit_transform(X['race/ethnicity'])
        
        self.encoder_par_edu_new = LabelEncoder()
        par_edu_new_encoded = self.encoder_par_edu_new.fit_transform(X['parental education'])
        
        self.encoder_gender = LabelEncoder()
        gender_encoded = self.encoder_gender.fit_transform(X['gender'])
        
        X["parental education encoded"] = par_edu_encoded
        X["test preparation course encoded"] = test_prep_encoded
        X["lunch encoded"] = lunch_encoded
        X["ethnicity encoded"] = ethn_encoded
        X["parental education new encoded"] = par_edu_new_encoded
        X["gender encoded"] = gender_encoded
        
        
        return X.drop(["parental level of education", "test preparation course", "lunch", 
                       "race/ethnicity", "parental education","gender"], axis=1)
    
catagorical_cols = ["parental level of education", "test preparation course", "lunch", 
                       "race/ethnicity", "parental education","gender"]
group_encode = group_encoder()

data_prep = Pipeline([("dataframe-selecter",DataFrameSelector(catagorical_cols)),
                      ("encoder",group_encode)])

perf_prep = data_prep.fit_transform(perf)
perf_prep.shape
# Dictionary of codes for every row that's encoded
parental_education_codes     = group_encode.encoder_par_edu.classes_
gender_codes                 = group_encode.encoder_gender.classes_
ethnicity_codes              = group_encode.encoder_ethn.classes_
test_preparation_codes       = group_encode.encoder_test_prep.classes_
lunch_codes                  = group_encode.encoder_lunch.classes_
parental_education_new_codes = group_encode.encoder_par_edu_new.classes_
# Adding numerical columns to prepared data
num_cols = np.array(perf[["math score","reading score","writing score","complete score"]])

perf_prep_total = np.concatenate((perf_prep, num_cols), axis=1)

perf_prep_total.shape
new_cols = ["parental level of education","test preperation","lunch",
            "race/ethniciy","parental education","gender","math score",
            "reading score","writing score","complete score"]
perf_new = pd.DataFrame(perf_prep_total, columns=new_cols)
perf_new.head()
corr_perf = perf_new.corr()
sns.heatmap(corr_perf)
corr_perf["lunch"]