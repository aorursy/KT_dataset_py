import keras
keras.__version__
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LeaveOneOut, KFold 
from keras import models, layers, losses, metrics, optimizers
PATH = "../input"
!ls {PATH}
df = pd.read_csv(f'{PATH}/spam_or_not_spam.csv')
df.head()
X = df[['email']].copy()
X.fillna("", inplace=True)

y = df[['label']].copy()
preprocess = make_column_transformer(
    (TfidfVectorizer(), 'email')
)
pipeline = make_pipeline(preprocess)
X_train, X_test, y_train, y_test = train_test_split(pipeline.fit_transform(X), y, test_size=0.2)
cross_validator = KFold(3, True)
def createNeuralNet(input_size):    
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(input_size,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
    
    return model
feature_count = X_train.shape[1]
train_history = createNeuralNet(feature_count)
for train_index, test_index in cross_validator.split(X_train.toarray(), y_train):
    result = train_history.fit(
                  X_train.toarray()[train_index],
                  y_train.values[train_index],
                  epochs=25,
                  batch_size=256,
                  validation_data=(X_train.toarray()[test_index], y_train.values[test_index]))
train_history.evaluate(X_test, y_test)