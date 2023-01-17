from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense
def skipper(fname):

    with open(fname) as fin:

        no_comments = (line for line in fin if not line.lstrip().startswith('#'))

        next(no_comments, None) # skip header

        for row in no_comments:

            yield row



dataset = np.loadtxt(skipper('/kaggle/input/pima-indians-diabetes-database/diabetes.csv'), delimiter=',')

X = dataset[:, 0:8]

Y = dataset[:, 8]



print(Y)
model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

model.fit(X,Y, epochs = 150, batch_size = 10)
_, accuracy = model.evaluate(X,Y)

print(accuracy)