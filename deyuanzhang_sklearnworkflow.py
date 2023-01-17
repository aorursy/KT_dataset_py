#The third person's information
A = [160, 50]
#The first person's information
B = [168, 80]
#The second person's information
C = [180, 80]
#The fourth person's information
D = [174, 62]

E = [165, 47]
persons = [A, B, C, D, E]

genders= ['F', 'M', 'M', 'F', 'F']

X_train = persons
Y_train = genders
print(X_train)
print(Y_train)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)
model = LogisticRegression()
model.fit(X_train, Y_train)
model = DecisionTreeClassifier()

model.fit(X_train, Y_train)
model = SVC()
model.fit(X_train, Y_train)
X_test = []
X_test.append([170, 50])
X_test.append([180, 80])
X_test.append([160, 40])
print(X_test)

Y_test = model.predict(X_test)
print(Y_test)
Y_real = ['F', 'M', 'M']

from sklearn.metrics import accuracy_score

acc = accuracy_score(Y_real, Y_test)
print(acc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


persons = []
gender = []
#The third person's information
persons.append([160, 50])
gender.append('F')
#The first person's information
persons.append([168, 80])
gender.append('M')
#The second person's information
persons.append([180, 80])
gender.append("M")

#The fourth person's information
persons.append([174, 62])
gender.append('F')

persons.append([165, 47])
gender.append('F')

X_train = persons
Y_train = gender

model = KNeighborsClassifier(n_neighbors = 1)
#model = DecisionTreeClassifier()


model.fit(X_train, Y_train)

X_test = []
X_test.append([170, 50])
X_test.append([180, 80])
X_test.append([160, 40])

Y_test = model.predict(X_test)

Y_real = ['F', 'M', 'M']
acc = accuracy_score(Y_real, Y_test)
print(acc)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


persons = []
genders = []
#The third person's information
persons.append([160, 50])
genders.append('F')
#The first person's information
persons.append([168, 80])
genders.append('M')
#The second person's information
persons.append([180, 80])
genders.append("M")

#The fourth person's information
persons.append([174, 62])
genders.append('F')

persons.append([165, 47])
genders.append('F')

persons.append([170, 47])
genders.append('F')

X_train = persons
Y_train = genders

model = SVC()

model.fit(X_train, Y_train)

X_test = []
X_test.append([170, 50])
X_test.append([180, 80])
X_test.append([160, 40])

Y_test = model.predict(X_test)

Y_real = ['F', 'M', 'M']
acc = accuracy_score(Y_real, Y_test)
print(acc)
