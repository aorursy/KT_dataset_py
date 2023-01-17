import json
import pandas as pd

with open('../input/train.json', 'r') as iodata:
    data = json.load(iodata)
    dataset = pd.DataFrame(data)
    
dataset.head()
from functools import reduce
import numpy as np

# Giả sử một texts có 3 câu văn là các phần tử trong list như bên dưới
texts = [['i', 'have', 'a', 'cat'], 
        ['he', 'have', 'a', 'dog'], 
        ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(reduce(lambda x, y: x + y, texts))))
# Dictionary sẽ chứa toàn bộ các từ của texts.

def bag_of_word(sentence):
    # Khởi tạo một vector có độ dài bằng với từ điển.
    vector = np.zeros(len(dictionary))
    # Đếm các từ trong một câu xuất hiện trong từ điển.
    for i, word in dictionary:
        count = 0
        # Đếm số từ xuất hiện trong một câu.
        for w in sentence:
            if w == word:
                count += 1
        vector[i] = count
    return vector
            
for i in texts:
    print(bag_of_word(i))
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range = (1, 1))
vect.fit_transform(['you have no dog', 'no, you have dog']).toarray()
vect.vocabulary_
vect = CountVectorizer(ngram_range = (1, 2))
vect.fit_transform(['you have no dog', 'no, you have dog']).toarray()
vect.vocabulary_
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean

vect = CountVectorizer(ngram_range = (3, 3), analyzer = 'char_wb')
n1, n2, n3, n4 = vect.fit_transform(['andersen', 'peterson', 'petrov', 'smith']).toarray()
euclidean(n1, n2), euclidean(n2, n3), euclidean(n3, n4)
# Cài đặt package pytesseract
import sys
!{sys.executable} -m pip install pytesseract
# Cài đặt terract
!{sys.executable} -m pip install tesseract
from pytesseract import image_to_string
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

##### Just a random picture from search
img = 'http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg'
img = requests.get(img)

img = Image.open(BytesIO(img.content))

# show image
img_arr = np.array(img)
plt.imshow(img_arr)

# img = Image.open(BytesIO(img.content))
# text = image_to_string(img)
# text
img2 = requests.get('https://photos.renthop.com/2/8393298_6acaf11f030217d05f3a5604b9a2f70f.jpg')
img2 = Image.open(BytesIO(img2.content))
img2 = np.array(img2)
plt.imshow(img2)
import sys
# install package reverse_geocoder
!{sys.executable} -m pip install reverse_geocoder
import reverse_geocoder as revgc
revgc.search((dataset.latitude[1], dataset.longitude[1]))
# dataset['created'].apply(lambda x: x.date().weekday())
from datetime import datetime

def parser(x):
    # Để biết được định dạng strftime của một chuỗi kí tự ta phải tra trong bàng string format time
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


dataset['created'] = dataset['created'].map(lambda x: parser(x))
#Kiểm tra định dạng time
for i, k in zip(dataset.columns, dataset.dtypes):
    print('{}: {}'.format(i, k))
dataset['weekday'] = dataset['created'].apply(lambda x: x.date().weekday())
print(dataset['weekday'].head())
dataset['is_weekend'] = dataset['created'].apply(lambda x: 1 if x.date().weekday() in [5, 6] else 0)
print(dataset['is_weekend'][:5])
# Download package user_agents
!{sys.executable} -m pip install user_agents
import user_agents
# Giả định có một user agent như bên dưới
ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36'
# Parser thông tin user agent
ua = user_agents.parse(ua)
# Khai thác các thuộc tính của user
print('Is a bot? ', ua.is_bot)
print('Is mobile? ', ua.is_mobile)
print('Is PC? ',ua.is_pc)
print('OS Family: ',ua.os.family)
print('OS Version: ',ua.os.version)
print('Browser Family: ',ua.browser.family)
print('Browser Version: ',ua.browser.version)
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta
from scipy.stats import shapiro
import statsmodels.api as sm 
import numpy as np

# Tạo ra một chuỗi phân phối beta
data = beta(1, 10).rvs(1000).reshape(-1, 1)
print('data shape:%s'%str(data.shape))
# Sử dụng kiểm định shapiro để kiểm tra tính phân phối chuẩn.
shapiro(data)
# Giá trị tới hạn và p-value
shapiro(StandardScaler().fit_transform(data))

# Giá trị tới hạn > p-value chúng ta sẽ bác bỏ giả thuyết về phân phối chuẩn.
# biến đổi dữ liệu theo phân phối chuẩn:
price = np.float64(dataset.price.values)
print('Head 5 of original prices:', price[:5])
price_std = StandardScaler().fit_transform(price.reshape(-1, 1))
print('Head 5 of standard scaling prices:\n', price_std[:5])
# Biến đổi trên tương đương với công thức sau:
price_std = (price - price.mean()) / price.std()
print('Head 5 of standard scaling prices:\n', price_std[:5])
from sklearn.preprocessing import MinMaxScaler
price_mm = MinMaxScaler().fit_transform(price.reshape(-1, 1))
print('Head of min max scaling price:\n', price_mm[:5])
# Hoặc đơn giản hơn là
price_mm = (price - price.min())/(price.max() - price.min())
print('Head of min max scaling price:\n', price_mm[:5])
# Tạo biến log price
price_log = np.log(price)
# Kiểm tra tính phân phối của các biến price, price_std, price_mm, price_log dựa trên biểu đồ Q-Q plot
sm.qqplot(price, loc = price.mean(), scale = price.std())
sm.qqplot(price_mm, loc = price_mm.mean(), scale = price_mm.std())
sm.qqplot(price_std, loc = price_std.mean(), scale = price_std.std())
sm.qqplot(price_log, loc = price_log.mean(), scale = price_log.std())
price_rm_outlier = price_log[(price_log < 12) & (price_log > 6)]
sm.qqplot(price_rm_outlier, loc = price_rm_outlier.mean(), scale = price_rm_outlier.std())
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

# Lấy dữ liệu example từ package sklearn
X, y = make_classification()

print('X: \n', X[:5, :5])
print('y: \n', y)
print('X shape:', X.shape)
print('y shape:', y.shape)
VarianceThreshold(.5).fit_transform(X).shape
VarianceThreshold(0.9).fit_transform(X).shape
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X_kbest = SelectKBest(f_classif, k = 5).fit_transform(X, y)
X_kvar = VarianceThreshold(0.9).fit_transform(X)
print('X shape after applying statistical selection: ',X_kbest.shape)
print('X shape after apply variance selection: ',X_kvar.shape)
# Hồi qui logistic
logit = LogisticRegression(solver='lbfgs', random_state=1)

# Cross validation cho:
# 1.dữ liệu gốc
acc_org = cross_val_score(logit, X, y, scoring = 'accuracy', cv = 5).mean()
# 2. Áp dụng phương sai
acc_var = cross_val_score(logit, X_kvar, y, scoring = 'accuracy', cv = 5).mean()
# 3. Áp dụng phương pháp thống kê
acc_stat = cross_val_score(logit, X_kbest, y, scoring = 'accuracy', cv = 5).mean()

print('Accuracy trên dữ liệu gốc:', acc_org)
print('Accuracy áp dụng phướng sai:', acc_var)
print('Accuracy dụng pp thống kê:', acc_stat)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Hồi qui theo RandomForest
rdFrt = RandomForestClassifier(n_estimators = 10, random_state = 1)
# Hồi qui theo LinearSVC
lnSVC = LinearSVC(C=0.01, penalty="l1", dual=False)
# Tạo một pipeline thực hiện lựa chọn biến từ RandomForest model và hồi qui theo logit
pipe1 = make_pipeline(StandardScaler(), SelectFromModel(estimator = rdFrt), logit)
# Tạo một pipeline thực hiện lựa chọn biến từ Linear SVC model và hồi qui theo logit
pipe2 = make_pipeline(StandardScaler(), SelectFromModel(estimator = lnSVC), logit)
# Cross validate đối với 
# 1. Mô hình logit
acc_log = cross_val_score(logit, X, y, scoring = 'accuracy', cv = 5).mean()
# 2. Mô hình RandomForest
acc_rdf = cross_val_score(rdFrt, X, y, scoring = 'accuracy', cv = 5).mean()
# 3. Mô hình pipe1
acc_pip1 = cross_val_score(pipe1, X, y, scoring = 'accuracy', cv = 5).mean()
# 3. Mô hình pipe2
acc_pip2 = cross_val_score(pipe2, X, y, scoring = 'accuracy', cv = 5).mean()

print('Accuracy theo logit:', acc_log)
print('Accuracy theo random forest:', acc_rdf)
print('Accuracy theo pipeline 1:', acc_pip1)
print('Accuracy theo pipeline 2:', acc_pip2)
!{sys.executable} -m pip install mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(logit, scoring = 'accuracy', 
                                     verbose = 2, 
                                     k_features = 3,
                                     forward = False,
                                     n_jobs = -1)

selector.fit(X, y)