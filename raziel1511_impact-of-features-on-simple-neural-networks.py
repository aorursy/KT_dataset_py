# Import Dataset

data = pd.read_csv('train.csv')

df = pd.DataFrame(data)



# Replace Sex with 1(male) and 0(female)

df.replace({'male': 1, 'female': 0}, inplace=True)
# Shuffle Data 

df = df.sample(frac=1.0)



# Delete Cabin (Data to sparse) and ID (pure Random)

# print(df['Survived'].isnull().sum())  # Age:177, Cabin: 687, Embarked:2

df.drop(['PassengerId', 'Cabin', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



# Create train and test

train_data = df.values[:800]

test_data = df.values[800:]



x_train = train_data[:, 1:]

y_train = np_utils.to_categorical(train_data[:, 0])



x_test = test_data[:, 1:]

y_test = np_utils.to_categorical(test_data[:, 0])



# Setup the Network

model = Sequential()

model.add(Dense(1500,

                activation='relu',

                input_shape=(x_train.shape[1],),

                kernel_regularizer=regularizers.l2(0.1)

                ))

model.add(Dropout(0.5))

model.add(Dense(2000, activation='relu',

                kernel_regularizer=regularizers.l2(0.1)

                ))

model.add(Dropout(0.5))

model.add(Dense(1500, activation='relu'))

model.add(Dense(2, activation='softmax',

                kernel_regularizer=regularizers.l2(0.1)

                ))



# Compile the model

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



tb = TensorBoard(log_dir='logs/{}'.format(time()))



# Train

model.fit(x=x_train, y=y_train, batch_size=200, verbose=2, epochs=25, callbacks=[tb])



# Eval

score = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy: {}".format(score[1]))

# Extract title from the name



def getTitle(name):

    '''

    :param name: Name of the format Firstname, Title Surename

    :return: Title

    '''



    m = re.search('(?<=,\s)\w+', name)

    return (m.group(0))





# Extract titles and count

title_set = {}

for name in df['Name']:

    title = getTitle(name)

    if title in title_set:

        title_set[title] = title_set[title] + 1

    else:

        title_set[title] = 1

print(title_set) 

# Output:

# {'Mr': 517, 'Ms': 1, 'Don': 1, 'the': 1, 

# 'Mlle': 2, 'Jonkheer': 1, 'Rev': 6, 

# 'Dr': 7, 'Miss': 182, 'Major': 2, 

# 'Sir': 1, 'Lady': 1, 'Mme': 1, 'Mrs': 125, 

# 'Master': 40, 'Col': 2, 'Capt': 1}

def getTitleNum(name):

    '''

    Assign a numeral according to the title

    :param name: 

    :return: numeral according to title

    '''



    title = getTitle(str(name).upper())



    title_dict = {}

    title_dict["MR"] = 0

    title_dict["MRS"] = 1

    title_dict["COL"] = 2

    title_dict["CAPT"] = 2

    title_dict["MAJOR"] = 2

    title_dict["MME"] = 3

    title_dict["MLLE"] = 3

    title_dict["MS"] = 3

    title_dict["MISS"] = 3

    title_dict["LADY"] = 4

    title_dict["SIR"] = 4

    title_dict["THE"] = 4

    title_dict["MASTER"] = 5

    title_dict["REV"] = 6

    title_dict["DR"] = 7



    if title in title_dict:

        return title_dict[title]

    else:

        return -1





df['Title'] = df.apply(lambda row: getTitleNum(row['Name']), axis=1)
# Discretize the Age

def discretAge(age):

    if age < 3:

        return 0

    if age < 12:

        return 1

    if age < 17:

        return 2

    if age < 50:

        return 3

    if age >= 50:

        return 4

    # Keep the missing values

    return age





df['DisAge'] = df.apply(lambda row: discretAge(row['Age']), axis=1)



# Replace Sex with 1(male) and 0(female)

df.replace({'male': 1, 'female': 0}, inplace=True)



# Shuffle Data and extract rows with missing age

df = df.sample(frac=1.0)

age_missing = df[df['Age'].isnull()]

age_complete = df[df['Age'].notnull()]





# Create train and test

ages = age_complete['DisAge'].values



x_train = age_complete[['Title','Pclass', 'Sex']].values[:650]

y_train = np_utils.to_categorical(ages[:650])



x_test = age_complete[['Title','Pclass', 'Sex']].values[650:]

y_test = np_utils.to_categorical(ages[650:])



# Setup the Network

model = Sequential()

model.add(Dense(800,

                activation='relu',

                input_shape=(x_train.shape[1],),

                kernel_regularizer=regularizers.l2(0.1)

                ))



model.add(Dropout(0.5))

model.add(Dense(800, activation='relu'))

model.add(Dense(5, activation='softmax',

                kernel_regularizer=regularizers.l2(0.1)

                ))



# Compile the model

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



tb = TensorBoard(log_dir='logs/{}'.format(time()))



# Train

model.fit(x=x_train, y=y_train, batch_size=200, verbose=2, epochs=25, callbacks=[tb])



# Eval

score = model.evaluate(x_test, y_test, verbose=0) # ~75%
# Apply the new model to predict the missing values



def predictAge(row):

    '''

    Use the trained network to predict the discrete Age

    :param row:

    :return:

    '''



    if math.isnan(float(row['DisAge'])):

        v = np.array(row[['Title', 'Pclass', 'Sex']].values)

        pred = model_age_prediction.predict(v.reshape((1,3)))

        return np.argmax(pred)

    return row['DisAge']



df['DisAge'] = df.apply(lambda row: predictAge(row), axis=1)