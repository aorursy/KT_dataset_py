import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bs4 import BeautifulSoup
import nltk #Natural Language Toolkit, for stop words
import re # get letter filter
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
train_path = '../input/labeledTrainData.tsv'
input_data = pd.read_csv(train_path, header= 0, delimiter= '\t',quoting= 3)
# print(input_data.review[0])
# print(input_data.describe())
input_data.describe()
input_data.columns
# Any results you write to the current directory are saved as output.
set(stopwords.words('english'))
def review_to_words(origin_data):
    data_no_html = BeautifulSoup(origin_data,'lxml').get_text()
    letters_only = re.sub('[^a-zA-Z]',' ',data_no_html)
    words=letters_only.lower().split()
    stops=set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))

clean_train_reviews = []
for i in range(0, input_data['review'].size):
    clean_train_reviews.append(review_to_words(input_data['review'][i]))
train_X = np.array(clean_train_reviews)
train_y = np.array(input_data['sentiment'])
from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(analyzer = 'word', max_features = 5000) 
train_data_features = vectorizer.fit_transform(train_X)
print(train_data_features.dtype) 
train_data_features = train_data_features.toarray() 
print(train_data_features.dtype) 
# vocab = vectorizer.get_feature_names()
# dist=np.sum(train_data_features,axis=0)
# for tag,count in zip(vocab,dist):
#     print(count,tag)
test = pd.read_csv("../input/testData.tsv", header=0, \
                    delimiter="\t", quoting=3)
clean_test_reviews = []
for i in range(0, test["review"].size ):
    clean_test_reviews.append(review_to_words(test["review"][i]))
test_X = np.array(clean_test_reviews)

test_data_features = vectorizer.fit_transform(test_X)
test_data_features = test_data_features.toarray() 
# vocab = vectorizer.get_feature_names()
# dist=np.sum(train_data_features,axis=0)
# for tag,count in zip(vocab,dist):
#     print(count,tag)
t_X, v_X, t_y, v_y = train_test_split(train_data_features, train_y, random_state = 0)

r_forest = RandomForestClassifier(n_estimators = 100) 
r_forest.fit(t_X, t_y)
predict_r = r_forest.predict(v_X)

d_model = DecisionTreeRegressor()
d_model.fit(t_X, t_y)
predict_d = d_model.predict(v_X)

result = pd.DataFrame({'validate': v_y, 'predictR': predict_r, 'predictD': predict_d})
result['DifferenceR'] = result['validate'] == result ['predictR']
result['DifferenceD'] = result['validate'] == result ['predictD']
result
print("Error rate R: " + str(len(result[result['DifferenceR'] == False]) / len(result)))
print("Error rate D: " + str(len(result[result['DifferenceD'] == False]) / len(result)))
r_forest.fit(train_data_features, train_y)
answer = r_forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":answer} )
output.to_csv( "Bag_of_Words_model1.csv", index=False, quoting=3)

# result=pd.DataFrame({ 'PassengerId': test_data.PassengerId, 'Survived': test_result })
# result.head()
# result.to_csv("Titanic_result.csv", index=False)
output.head()
