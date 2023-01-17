import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense
#data preprocessing

df = pd.read_csv('../input/HR_comma_sep.csv')

X = df.ix[:, df.columns != 'left']

X = X.iloc[:, 0:9].values

y = df.iloc[:, -4].values

# label encoding

le = LabelEncoder()

X[:, -1] = le.fit_transform(X[:, -1])

X[:, -2] = le.fit_transform(X[:, -2])

# One hot encoding

ohe = OneHotEncoder(categorical_features=[1])

X = ohe.fit_transform(X).toarray()

# Creating sets

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Feature scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# creating the model

clf = Sequential([

    Dense(units=11, kernel_initializer='uniform', activation='relu', input_dim=10),

    Dense(units=11, kernel_initializer='uniform', activation='relu'), # units are based on my creativity :3

    Dense(1, kernel_initializer='uniform', activation='sigmoid') #output

])
# compiling the model

clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
clf.fit(X_train, Y_train, batch_size=9, epochs=10) # less training due to slow kaggle servers
score = clf.evaluate(X_test, Y_test, batch_size=128)

print(score[1]*100, '%') # 96.9333337148% or 0.96133