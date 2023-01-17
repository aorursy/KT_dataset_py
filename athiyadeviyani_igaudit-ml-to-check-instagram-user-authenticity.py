# Imports



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier



from instagram_private_api import Client, ClientCompatPatch

import getpass



import random
train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")
train.head()
train.describe()
train.info()
train.shape
test.head()
test.describe()
test.info()
test.shape
print(train.isna().values.any().sum())

print(test.isna().values.any().sum())
fig, ax = plt.subplots(figsize=(15,10))  

corr=train.corr()

sns.heatmap(corr, annot=True)
# Labels

train_Y = train.fake

train_Y = pd.DataFrame(train_Y)



# Data

train_X = train.drop(columns='fake')

train_X.head()
# Labels

test_Y = test.fake

test_Y = pd.DataFrame(test_Y)



# Data

test_X = test.drop(columns='fake')

test_X.head()
# Baseline classifier

fakes = len([i for i in train.fake if i==1])

auth = len([i for i in train.fake if i==0])

fakes, auth



# classify everything as fake

pred = [1 for i in range(len(test_X))]

pred = np.array(pred)

print("Baseline accuracy: " + str(accuracy_score(pred, test_Y)))
# Statistical method

def stat_predict(test_X, r):

    pred = []

    for row in range(len(test_X)):   

        followers = test_X.loc[row]['#followers']

        followings = test_X.loc[row]['#follows']

        if followers == 0:

            followers = 1

        if followings == 0:

            followings == 1



        ratio = followings/followers



        if ratio >= r:

            pred.append(1)

        else:

            pred.append(0)

    

    return np.array(pred)

accuracies = []

for i in [x / 10.0 for x in range(5, 255, 5)]:

    prediction = stat_predict(test_X, i)

    accuracies.append(accuracy_score(prediction, test_Y))



f, ax = plt.subplots(figsize=(20,10))

plt.plot([x / 10.0 for x in range(5, 255, 5)], accuracies)

plt.plot([2.5 for i in range(len(accuracies))], accuracies, color='red')

plt.title("Accuracy for different thresholds", size=30)

plt.xlabel('Ratio', fontsize=20)

plt.ylabel('Accuracy', fontsize=20)

print("Maximum Accuracy for the statistical method: " + str(max(accuracies)))
lm = LogisticRegression()



# Train the model

model1 = lm.fit(train_X, train_Y)



# Make a prediction

lm_predict = model1.predict(test_X)
# Compute the accuracy of the model

acc = accuracy_score(lm_predict, test_Y)

print("Logistic Regression accuracy: " + str(acc))
accuracies = []



# Compare the accuracies of using the KNN classifier with different number of neighbors

for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=i)

    model_2 = knn.fit(train_X,train_Y)

    knn_predict = model_2.predict(test_X)

    accuracy = accuracy_score(knn_predict,test_Y)

    accuracies.append(accuracy)



max_acc = (0, 0)

for i in range(1, 10):

    if accuracies[i-1] > max_acc[1]:

        max_acc = (i, accuracies[i-1])



max_acc



f, ax = plt.subplots(figsize=(20,10))

plt.plot([i for i in range(1,10)], accuracies)

plt.plot([7 for i in range(len(accuracies))], accuracies, color='red')

plt.title("Accuracy for different n-neighbors", size=30)

plt.xlabel('Number of neighbors', fontsize=20)

plt.ylabel('Accuracy', fontsize=20)



print("The highest accuracy obtained using KNN is " + str(max_acc[1]) + " achieved by a value of n=" + str(max_acc[0]))
DT = DecisionTreeClassifier()



# Train the model

model3 = DT.fit(train_X, train_Y)



# Make a prediction

DT_predict = model3.predict(test_X)
# Compute the accuracy of the model

acc = accuracy_score(DT_predict, test_Y)

print("Decision Tree accuracy: " + str(acc))
rfc = RandomForestClassifier()



# Train the model

model_4 = rfc.fit(train_X, train_Y)



# Make a prediction

rfc_predict = model_4.predict(test_X)
# Compute the accuracy of the model

acc = accuracy_score(rfc_predict, test_Y)

print("Random Forest accuracy: " + str(acc))
def login():

    username = input("username: ")

    password = getpass.getpass("password: ")

    api = Client(username, password)

    return api



api = login()
def get_ID(username):

    return api.username_info(username)['user']['pk']
# The user used for the experiment below is anonymised!

# i.e. this cell was run and then changed to protect the user's anonymity

userID = get_ID('<USERNAME HERE>') 
rank = api.generate_uuid()
def get_followers(userID, rank):

    followers = []

    next_max_id = True

    

    while next_max_id:

        if next_max_id == True: next_max_id=''

        f = api.user_followers(userID, rank, max_id=next_max_id)

        followers.extend(f.get('users', []))

        next_max_id = f.get('next_max_id', '')

    

    user_fer = [dic['username'] for dic in followers]

    

    return user_fer
followers = get_followers(userID, rank)
# You can check the number of followers if you'd like to

# len(followers)
# This will print the first follower username on the list

# print(followers[0])
# This will get the information on a certain user

info = api.user_info(get_ID(followers[0]))['user']



# Check what information is available for one particular user

info.keys()
def get_data(info):

    

    """Extract the information from the returned JSON.

    

    This function will return the following array:

        data = [profile pic,

                nums/length username,

                full name words,

                nums/length full name,

                name==username,

                description length,

                external URL,

                private,

                #posts,

                #followers,

                #followings]

    """

    

    data = []

    

    # Does the user have a profile photo?

    profile_pic = not info['has_anonymous_profile_picture']

    if profile_pic == True:

        profile_pic = 1

    else:

        profile_pic = 0

    data.append(profile_pic)

    

    # Ratio of number of numerical chars in username to its length

    username = info['username']

    uname_ratio = len([x for x in username if x.isdigit()]) / float(len(username))

    data.append(uname_ratio)

    

    # Full name in word tokens

    full_name = info['full_name']

    fname_tokens = len(full_name.split(' '))

    data.append(fname_tokens)

    

    # Ratio of number of numerical characters in full name to its length

    if len(full_name) == 0:

        fname_ratio = 0

    else:

        fname_ratio = len([x for x in full_name if x.isdigit()]) / float(len(full_name))

    data.append(fname_ratio)

    

    # Is name == username?

    name_eq_uname = (full_name == username)

    if name_eq_uname == True:

        name_eq_uname = 1

    else:

        name_eq_uname = 0

    data.append(name_eq_uname)

    

    # Number of characters on user bio 

    bio_length = len(info['biography'])

    data.append(bio_length)

    

    # Does the user have an external URL?

    ext_url = info['external_url'] != ''

    if ext_url == True:

        ext_url = 1

    else:

        ext_url = 0

    data.append(ext_url)

    

    # Is the user private or no?

    private = info['is_private']

    if private == True:

        private = 1

    else:

        private = 0

    data.append(private)

    

    # Number of posts

    posts = info['media_count']

    data.append(posts)

    

    # Number of followers

    followers = info['follower_count']

    data.append(followers)

    

    # Number of followings

    followings = info['following_count']

    data.append(followings)

    

  

    return data
# Check if the function returns as expected

get_data(info)
# Get a random sample of 50 followers

random_followers = random.sample(followers, 50)
f_infos = []



for follower in random_followers:

    info = api.user_info(get_ID(follower))['user']

    f_infos.append(info)
f_table = []



for info in f_infos:

    f_table.append(get_data(info))

    

f_table
test_data = pd.DataFrame(f_table,

                         columns = ['profile pic', 

                                    'nums/length username', 

                                    'fullname words',

                                    'nums/length fullname',

                                    'name==username',

                                    'description length',

                                    'external URL',

                                    'private',

                                    '#posts',

                                    '#followers',

                                    '#follows'])

test_data
rfc = RandomForestClassifier()



# Train the model

# We've done this in Part 2 but I'm redoing it here for coherence ☺️

rfc_model = rfc.fit(train_X, train_Y)
rfc_labels = rfc_model.predict(test_data)

rfc_labels
no_fakes = len([x for x in rfc_labels if x==1])
authenticity = (len(random_followers) - no_fakes) * 100 / len(random_followers)

print("User X's Instagram Followers is " + str(authenticity) + "% authentic.")
def get_user_posts(userID, min_posts_to_be_retrieved):

    # Retrieve all posts from my profile

    my_posts = []

    has_more_posts = True

    max_id = ''

    

    while has_more_posts:

        feed = api.user_feed(userID, max_id=max_id)

        if feed.get('more_available') is not True:

            has_more_posts = False 

            

        max_id = feed.get('next_max_id', '')

        my_posts.extend(feed.get('items'))

        

        # time.sleep(2) to avoid flooding

        

        if len(my_posts) > min_posts_to_be_retrieved:

            print('Total posts retrieved: ' + str(len(my_posts)))

            return my_posts

            

        if has_more_posts:

            print(str(len(my_posts)) + ' posts retrieved so far...')

           

    print('Total posts retrieved: ' + str(len(my_posts)))

    

    return my_posts
posts = get_user_posts(userID, 10)
random_post = random.sample(posts, 1)
random_post[0].keys()
likers = api.media_likers(random_post[0]['id'])
likers_usernames = [liker['username'] for liker in likers['users']]
random_likers = random.sample(likers_usernames, 50)
l_infos = []



for liker in random_likers:

    info = api.user_info(get_ID(liker))['user']

    l_infos.append(info)
l_table = []



for info in l_infos:

    l_table.append(get_data(info))



l_table
# Generate pandas dataframe 

l_test_data = pd.DataFrame(l_table,

                         columns = ['profile pic', 

                                    'nums/length username', 

                                    'fullname words',

                                    'nums/length fullname',

                                    'name==username',

                                    'description length',

                                    'external URL',

                                    'private',

                                    '#posts',

                                    '#followers',

                                    '#follows'])

l_test_data
rfc = RandomForestClassifier()

rfc_model = rfc.fit(train_X, train_Y)

rfc_labels_likes = rfc_model.predict(l_test_data)

rfc_labels_likes
no_fake_likes = len([x for x in rfc_labels_likes if x==1])
media_authenticity = (len(random_likers) - no_fake_likes) * 100 / len(random_likers)

print("The media with the ID:XXXXX has " + str(media_authenticity) + "% authentic likes.")
# Re-login because of API call limits 

api = login()
userID_y = get_ID('<USERNAME>')
rank = api.generate_uuid()
y_followers = get_followers(userID_y, rank)
y_random_followers = random.sample(y_followers, 50)
y_infos = []



for follower in y_random_followers:

    info = api.user_info(get_ID(follower))['user']

    y_infos.append(info)
y_table = []



for info in y_infos:

    y_table.append(get_data(info))

    

y_table
# Generate pandas dataframe 

y_test_data = pd.DataFrame(y_table,

                         columns = ['profile pic', 

                                    'nums/length username', 

                                    'fullname words',

                                    'nums/length fullname',

                                    'name==username',

                                    'description length',

                                    'external URL',

                                    'private',

                                    '#posts',

                                    '#followers',

                                    '#follows'])

y_test_data
# Predict (no retraining!)

rfc_labels_y = rfc_model.predict(y_test_data)

rfc_labels_y
# Calculate the number of fake accounts in the random sample of 50 followers

no_fakes_y = len([x for x in rfc_labels_y if x==1])
# Calculate the authenticity

y_authenticity = (len(y_random_followers) - no_fakes_y) * 100 / len(y_random_followers)

print("User Y's Instagram Followers is " + str(y_authenticity) + "% authentic.")
y_posts = get_user_posts(userID_y, 10)
y_random_post = random.sample(y_posts, 1)
y_likers = api.media_likers(y_random_post[0]['id'])
y_likers_usernames = [liker['username'] for liker in y_likers['users']]
y_random_likers = random.sample(y_likers_usernames, 50)
y_likers_infos = []



for liker in y_random_likers:

    info = api.user_info(get_ID(liker))['user']

    y_likers_infos.append(info)
y_likers_table = []



for info in y_likers_infos:

    y_likers_table.append(get_data(info))

    

y_likers_table
y_likers_data = pd.DataFrame(y_likers_table,

                         columns = ['profile pic', 

                                    'nums/length username', 

                                    'fullname words',

                                    'nums/length fullname',

                                    'name==username',

                                    'description length',

                                    'external URL',

                                    'private',

                                    '#posts',

                                    '#followers',

                                    '#follows'])

y_likers_data
# Predict!

y_likers_pred = rfc_model.predict(y_likers_data)

y_likers_pred
# Calculate the number of fake likes

no_fakes_yl = len([x for x in y_likers_pred if x==1])



# Calculate media likes authenticity

y_post_authenticity = (len(y_random_likers) - no_fakes_yl) * 100 / len(y_random_likers)

print("The media with the ID:YYYYY has " + str(y_post_authenticity) + "% authentic likes.")
y_posts[0].keys()
count = 0



for post in y_posts:

    count += post['comment_count']

    count += post['like_count']

    

average_engagements = count / len(y_posts)

engagement_rate = average_engagements*100 / len(y_followers)



engagement_rate