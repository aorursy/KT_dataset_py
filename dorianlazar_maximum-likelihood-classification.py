import numpy as np

import pandas as pd
df = pd.read_csv('../input/titanic/train.csv')
df
df.describe()
def simplify(df: pd.DataFrame):

    del df['PassengerId']

    del df['Name']

    del df['Pclass']

    df['Sex'] = (df['Sex'].values == 'male').astype(int)

    mean_age = np.mean(df['Age'].values[~np.isnan(df['Age'].values)])

    df['Age'] = [mean_age if np.isnan(age) else age for age in df['Age'].values]

    del df['Ticket']

    del df['Cabin']

    mean_fare = np.mean(df['Fare'].values[~np.isnan(df['Fare'].values)])

    df['Fare'] = [mean_fare if np.isnan(fare) else fare for fare in df['Fare'].values]

    df['S'] = (df['Embarked'].values == 'S').astype(int) + df['Embarked'].isna().values.astype(int)

    del df['Embarked']
labels = df['Survived'].values

del df['Survived']



simplify(df)
df
df.describe()
def train_test_split(x: np.ndarray, y: np.ndarray, train_ratio: float) -> tuple:

    '''

    Returns: tuple of form (x_train, y_train, x_test, y_test)

    '''

    n = x.shape[0]

    train_size = int(n * train_ratio)

    

    train_indices = np.random.choice(n, train_size)

    test_indices = [i for i in np.arange(n) if i not in train_indices]

    

    x_train = np.array([x[i] for i in train_indices])

    y_train = np.array([y[i] for i in train_indices])

    x_test = np.array([x[i] for i in test_indices])

    y_test = np.array([y[i] for i in test_indices])

    

    return (x_train, y_train, x_test, y_test)
class MLClassifier:

    def fit(self, x: np.ndarray, y: np.ndarray):

        self.d = x.shape[1] # no. of variables / dimensions

        self.nclasses = len(set(y))

        

        self.mu_list = []

        self.sigma_list = []

        

        n = x.shape[0] # no. of observations

        for i in range(self.nclasses):

            cls_x = np.array([x[j] for j in range(n) if y[j] == i])

            mu = np.mean(cls_x, axis=0)

            sigma = np.cov(cls_x, rowvar=False)

            self.mu_list.append(mu)

            self.sigma_list.append(sigma)

    

    def _class_likelihood(self, x: np.ndarray, cls: int) -> float:

        mu = self.mu_list[cls]

        sigma = self.sigma_list[cls]

        if np.sum(np.linalg.eigvals(sigma) <= 0) != 0:

            print(f'Warning! Covariance matrix for label {cls} is not positive definite!\n')

            print('The predicted likelihood will be 0.')

            return 0.0

        d = self.d

        

        exp = (-1/2)*np.dot(np.matmul(x-mu, np.linalg.inv(sigma)), x-mu)

        s_val = ((2*np.pi)**d)*np.linalg.det(sigma)

        c = 1/np.sqrt(s_val)

        

        return c * (np.e**exp)

    

    def predict(self, x: np.ndarray) -> int:

        likelihoods = [self._class_likelihood(x, i) for i in range(self.nclasses)]

        return np.argmax(likelihoods)

    

    def score(self, x: np.ndarray, y: np.ndarray):

        n = x.shape[0]

        predicted_y = np.array([self.predict(x[i]) for i in range(n)])

        n_correct = np.sum(predicted_y == y)

        return n_correct/n
(x_train, y_train, x_test, y_test) = train_test_split(df.values, labels, 0.8)
mlc = MLClassifier()

mlc.fit(x_train, y_train)
mlc.score(x_test, y_test)
df_test = pd.read_csv('../input/titanic/test.csv')
df_test
df_test.describe()
df_output = pd.DataFrame(columns=['PassengerId', 'Survived'])
df_output['PassengerId'] = df_test['PassengerId'].values
df_output
simplify(df_test)
df_test
df_test.describe()
mlc_final = MLClassifier()

mlc_final.fit(df.values, labels)
survived = []

for row in df_test.iterrows():

    y = mlc_final.predict(row[1].values)

    survived.append(y)
df_output['Survived'] = survived
df_output
df_output.to_csv('predictions.csv', index=False)