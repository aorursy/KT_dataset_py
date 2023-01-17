from multiprocessing import Pool, Queue, Manager, cpu_count
from requests import Session
from time import sleep
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
directory = Manager().dict()  
q = Queue() 
data = []
proxies = []
sessions = [Session()] * 2
def get_clemson_data(session, person):
    try:
        data = session.get("https://my.clemson.edu/srv/feed/dynamic/directory/search?name=" + person.name)
        person.cuid = data.json()[0].get('cn')
        data = session.get('https://my.clemson.edu/srv/feed/dynamic/directory/getInfoByCN?cn=' + person.cuid)
        data = data.json()
        if person.name.split() != [data.get('name').get('first'), data.get('name').get('last')]:
            raise PersonNotAtClemson(person)
    except (KeyError, IndexError, AttributeError, URLError):
        raise PersonNotAtClemson(person)
    else:
        try:
            person.class_standing = data.get('student').get('class')
            person.major = data.get('student').get('major').get('name')
            with suppress(Exception):
                urlretrieve(data['photo_url'], 'output/images/' + person.cuid + '.png')
        except AttributeError:
            pass
def get_venmo_data(session, person, proxy):
    try:
        data = session.get("https://venmo.com/" + person.username, timeout=2, proxies={'https': proxy})
        soup = BeautifulSoup(data.text, 'lxml')
        person.friends = [friend['details'].strip(')').split(' (') for friend in soup(cardtype='profile')]
        if not person.friends:
            raise TooManyRequests if 'Sorry' in data.text else SessionTimeOut

        rgx = re.compile(r"^venmo.page_user.*?id\":\s([0-9]*)", flags=re.MULTILINE)
        person.id = re.search(rgx, data.text).group(1)
        data = session.get("https://venmo.com/api/v5/users/" + person.id + "/feed", timeout=2, proxies={'https':proxy})
        data = data.json()

        person.transactions = [
            [t['actor']['username'], t['transactions'][0]['target']['username'], t['message'], t['created_time']]
            for t in data['data']]
    except RequestException:
        raise NonFatalException(person)
def scrape(i):
    # exit if demand for queue is too high
    try:
        person = Person(q.get(timeout=5))
    except Exception:
        return

    # random proxy to avoid detection
    proxy = 'http://' + np.random.choice(proxies)

    try:
        if person.username in directory:
            raise PersonInDirectory(person)

        get_clemson_data(sessions[0], person)
        get_venmo_data(sessions[1], person, proxy=proxy)

    # exceptions in which individual is not scraped
    except NonFatalException as e:
        if isinstance(e, PersonInDirectory):
            pass
        elif isinstance(e, PersonNotAtClemson):
            directory[person.username] = False

    # exceptions that interrupt program flow
    except FatalException as e:
        if isinstance(e, TooManyRequests):
            pass
        # if session times out, program waits for updated cookies
        elif isinstance(e, SessionTimeOut):
            print("SLEEPING")
            sleep(900)
            load_settings()
        q.put([person.name, person.username])

    # catch other exceptions to prevent termination
    except Exception as e:
        print("UNKNOWN ERROR", e)

    # add person to directory and friends to queue
    else:
        directory[person.username] = True
        [q.put(friend) for friend in person.friends]
        return person.dump()
def run(n):
    print('Scraping started.')
    pool = Pool(cpu_count())
    # keep scraping until either queue is empty or goal is reached
    while not q.empty() and len(data) < n:
        data.extend(filter(None, pool.map(main, range(3000))))
        print("Number of people checked:{}".format(len(directory)))
        print("Number of students scraped:{}".format(len(data)))
        # save data at intervals
        with open('output/directory.pkl', 'wb') as f:
            pickle.dump(dict(directory), f)
        pd.DataFrame(data).to_csv('output/out.csv')
    pool.close()
    print('Scraping Finished')
original_df = pd.read_hdf('Data/student_data.hdf')
original_df.head()
df = original_df.copy()
directory = df['major'].to_dict()
df['friends'] = df['friends'].apply(lambda f: list(filter(None,map(directory.get, np.array(f)[:, 1]))))

tr = []                                              
for person, trans in zip(df.index, df['transactions']):
    tr.append(list(filter(None, map(directory.get, [t[0] if person is t[1] else t[1] for t in trans]))))
df['transactions'] = tr
 
df.head()
sns.set_style('ticks')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,7))
ax1.hist(df['transactions'].apply(len))
ax1.set(ylabel='Number of Students', xlabel='Transaction Count')

ax2.hist(df['friends'].apply(len), range=[0,100])
ax2.set(title='Data Distributions', xlabel="Number of Friends")

ax3.hist(df['major'].apply(len))
ax3.set(xlabel='Major Size')
list_of_majors = set(df['major'])
lookup = dict(zip(list_of_majors, np.eye(len(list_of_majors)))) #  major to sparse encoding
df = df.applymap(lambda x: lookup.get(x) if isinstance(x,str) else list(map(lookup.get, x))) # encode dataframe
df.head()
df = df[(df['friends'].apply(len) != 0) & (df['transactions'].apply(len) != 0)]
m = np.array([df['major'].tolist(), df['friends'].apply(sum).tolist(), df['transactions'].apply(sum).tolist()])
m
min_friends = 5
min_transactions = 5
min_major_size = 1

m = m[...,m[0].sum(0) >= min_major_size]
m = m[:,np.logical_and(np.logical_and(m[1].sum(1) >= min_friends,m[2].sum(1) >= min_transactions), m[0].sum(1) > 0),]
m = [x/x.sum(1, keepdims=1) for x in m]
train_y, test_y, train_x1, test_x1, train_x2, test_x2 = train_test_split(*m)
model_1 = Sequential([Dense(200, input_dim=train_x1.shape[1]), Dense(train_x1.shape[1]), Activation('softmax')])
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
model_1.fit(x=train_x1, y=train_y, epochs=15, batch_size=10, verbose=1)
def evaluate(preds):
    return np.mean(np.argmax(test_y, axis=1) == preds)
nn1_preds = np.argmax(model_1.predict(test_x1), axis=1)
evaluate(nn1_preds)
model_2 = Sequential([Dense(200, input_dim=train_x1.shape[1]), Dense(train_x1.shape[1]), Activation('softmax')])
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
model_2.fit(x=train_x2, y=train_y, epochs=15, batch_size=10, verbose=1)
nn2_preds = np.argmax(model_2.predict(test_x2), axis=1)
evaluate(nn2_preds)
baseline_1 = np.argmax(test_x1, axis=1)
baseline_2 = np.argmax(test_x2, axis=1)
evaluate(baseline_1), evaluate(baseline_2)
train_y_non_sparse = [np.argmax(y) for y in train_y]
def test_model(model, name):
    df = pd.DataFrame()
    model.fit(train_x1, train_y_non_sparse)
    df['x1'] = model.predict(test_x1)
    model.fit(train_x2, train_y_non_sparse)
    df['x2'] = model.predict(test_x2)
    df['Model'] = name
    return df
clf = LogisticRegression()
clf_preds = test_model(clf, 'Logistic Regression')
neigh = KNeighborsClassifier(n_neighbors = 10)
neigh_preds = test_model(neigh, 'K-Nearest Neigbors')
tree = DecisionTreeClassifier()
tree_preds = test_model(tree, 'Decision Tree')
baseline_preds = pd.DataFrame({'Model':'Baseline', 'x1':baseline_1, 'x2':baseline_2})
nn_preds = pd.DataFrame({'Model':'Neural Network', 'x1': nn1_preds, 'x2':nn2_preds})
results = pd.concat((baseline_preds, clf_preds, neigh_preds, tree_preds, nn_preds))
results = results.melt(id_vars='Model', value_vars=['x1', 'x2'], var_name='Dataset', value_name='Preds')
results
sns.set_style('ticks')
fig, ax = plt.subplots(figsize=(10, 8))
plt = sns.barplot(ax=ax, x='Model', y='Preds', data=results, hue='Dataset', estimator=evaluate, ci=None)
plt.set(title='Model Performance', ylabel='Accuracy')

# add accuracies to top of bars
for p in plt.patches:
    plt.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
graph_df = original_df.copy()
graph_df['friends'] = graph_df['friends'].apply(lambda f: list(filter(None,map(directory.get, np.array(f)[:, 1]))))
G = nx.Graph()
max_nodes = 100 # drawing demo purposes
for student, friends in zip(graph_df.index, graph_df['friends']):
    if G.number_of_nodes() >= max_nodes:
        break
    for friend in friends:
        G.add_edge(student, friend)
nx.draw(G)
nx.write_graphml(G,'Data/graph.graphml') # export graph