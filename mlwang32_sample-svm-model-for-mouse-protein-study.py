# This is a simple practice for building a classification model for c-CS-s vs. c-SC-s using support vector machine
# Graphs and preliminary statistics were generated with reference to def me(x)'s excellent R study
# Codes were executed using Python3

# Import Necessary Packages
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
mouse = pd.read_csv('../input/Data_Cortex_Nuclear.csv')
mouse.head()
# Check total missing values for each column
mouse.isnull().sum()
# Check missing values for a randomly selected protein 'DYRK1A_N'
mouse[mouse['DYRK1A_N'].isnull()]
# Show frequency of unique value for 'class'
mouse['class'].value_counts()
# Plot frequency of unique values for 'class'
mouse['class'].value_counts().plot(kind='bar')
plt.title('Class Frequency Table')
plt.xlabel('Class Type')
plt.xticks(rotation='horizontal')
plt.ylabel('Counts')
plt.show()
# Histogram of protein expression level - DYRK1A
protein = mouse['DYRK1A_N'].dropna()
plt.hist(protein, bins=25, edgecolor='black')
plt.title('Expression Frequency Table')
plt.xlabel('Protein Expression Level')
plt.ylabel('Counts')
plt.show()
# Boxplot of Protein Expression Level by Class
protein = mouse[['DYRK1A_N', 'class']].dropna()
sns.boxplot(x='class', y='DYRK1A_N', data=protein)
plt.show()
# One Way ANOVA Test
# Assumption 1: normal distribution for the variable in question
# Assumption 2: same population variance across all groups
# Assumption 3: independent observations
protein = mouse[['DYRK1A_N', 'class']].dropna()

g1 = protein[(protein['class'] == 'c-CS-m')]['DYRK1A_N']
g2 = protein[(protein['class'] == 'c-SC-m')]['DYRK1A_N']
g3 = protein[(protein['class'] == 'c-CS-s')]['DYRK1A_N']
g4 = protein[(protein['class'] == 'c-SC-s')]['DYRK1A_N']
g5 = protein[(protein['class'] == 't-CS-m')]['DYRK1A_N']
g6 = protein[(protein['class'] == 't-SC-m')]['DYRK1A_N']
g7 = protein[(protein['class'] == 't-CS-s')]['DYRK1A_N']
g8 = protein[(protein['class'] == 't-SC-s')]['DYRK1A_N']

print(stats.f_oneway(g1, g2, g3, g4, g5, g6, g7, g8)) 
# Pairwise T-Test Comparison of Means
print(stats.ttest_ind(g1, g2))
print(stats.ttest_ind(g5, g7))
# Data Cleaning
# Remove columns with NaN >= 10, then drop rows with missing data and categorical columns
mouse_new = mouse[mouse.columns[mouse.isnull().sum() < 10]].dropna().drop(['MouseID', 'Genotype', 'Treatment', 'Behavior'], axis=1)
mouse_new.shape
# Build a classification model for c-CS-s vs. c-SC-s using support vector machine
df = mouse_new[(mouse_new['class'] == 'c-CS-s') | (mouse_new['class'] == 'c-SC-s')].copy()
df.loc[:, 'class'].replace({'c-CS-s': 1, 'c-SC-s': 0}, inplace=True)
X = df.drop(['class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

svm = SVC(gamma='scale')
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))
