# Data handling and processing
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# NLP
import nltk
nltk.download('wordnet')
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



Division = 'All'
Department = 'All'
Class = 'All'
Age = 0
# filters
data = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
data.dropna(subset=['Review Text'])
data.dropna(inplace=True)
data = data[['Clothing ID','Age','Review Text','Division Name','Department Name','Class Name','Recommended IND','Rating']]
Division_list = list(set(data['Division Name']))
Department_list = list(set(data['Department Name']))
Class_list = list(set(data['Class Name']))

#print(Division_list)
#print(Department_list)
#print(Class_list)

Division_list.append("All")
Department_list.append("All")
Class_list.append("All")
print("Welcome to our store")
print("Tell us about yourself")
print("Please select the Division of the cloth you want")
for i in range(len(Division_list)):
    print(i+1,Division_list[i])
Div_index = int(input())-1
Division = Division_list[Div_index]
print("Please select the Department of the cloth you want")
for i in range(len(Department_list)):
    print(i+1,Department_list[i])
Dep_index = int(input())-1
Department = Department_list[Dep_index]
print("Please select the Class of the cloth you want")
for i in range(len(Class_list)):
    print(i+1,Class_list[i])
Class_index = int(input())-1
Class = Class_list[Class_index]

print("Please enter your age")
Age = int(input())


data1 = data
if Division != "All" :
    data1 = data1.loc[data1['Division Name'] == Division ]
if Department != "All" :
    data1 = data1.loc[data1['Department Name'] == Department ]
if Class != "All" :
    data1 = data1.loc[data1['Class Name'] == Class ]
if Age !=0 :
    data1 = data1.loc[data1['Age'] > Age - 2.5 ]
    data1 = data1.loc[data1['Age'] < Age + 2.5 ]
#print(data1)
#print(data)
#ID_list = list(set(data1['Clothing ID']))
#print(ID_list)


#data = data.loc[data['Clothing ID'] in ID_list]

#print(data)

stop = text.ENGLISH_STOP_WORDS

# Basic text cleaning function
def remove_noise(text):
    
    # Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    
    # Remove special characters
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    
    # Remove punctuation
    text = text.str.replace('[^\w\s]', '')
    
    # Remove numbers
    text = text.str.replace('\d+', '')
    
    # Remove Stopwords
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    # Convert to string
    text = text.astype(str)
        
    return text
data1['Filtered Review Text'] = remove_noise(data1['Review Text'])

# Defining a sentiment analyser function
def sentiment_analyser(text):
    return text.apply(lambda Text: pd.Series(TextBlob(Text).sentiment.polarity))

# Applying function to reviews
data1['Polarity'] = sentiment_analyser(data1['Filtered Review Text'])
data1.head(10)
# Visualising polarity between recommending and non-recommending customers, then getting value counts
g = sns.FacetGrid(data1, col="Recommended IND", col_order=[1, 0])
g = g.map(plt.hist, "Polarity", bins=20, color="g")

recommend = data1.groupby(['Recommended IND'])
recommend['Polarity'].mean()
# Visualizing Polarity and Rating
x_axis = list(data1['Rating'])
y_axis = list(data1['Polarity'])
plt.xlabel('Rating')
plt.ylabel('Polarity')
plt.scatter(x_axis, y_axis)
plt.show()

data11 = data1
while not data11.empty :
    max_polr = max(list(data11['Polarity']))
    #print(max_polr)
    max_polr_data = data11.loc[data11['Polarity'] == max_polr]
    print("Clothing ID : ",list(max_polr_data['Clothing ID'])[0])
    print("Department : ",list(max_polr_data['Department Name'])[0] )
    print("Division : ", list(max_polr_data['Division Name'])[0])
    print("Class : ", list(max_polr_data['Class Name'])[0])
    print(list(max_polr_data['Review Text'])[0])
    data11 = data11.loc[data11['Clothing ID'] != list(max_polr_data['Clothing ID'])[0] ]
    


