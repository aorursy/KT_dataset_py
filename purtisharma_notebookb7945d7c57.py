# Install the library
!pip install jovian --upgrade --quiet
# Import it
import jovian
project_name='python-practice-assignment'
# Capture and upload a snapshot
jovian.commit(project=project_name, privacy='secret', evironment=None)
name = 'Purti Sharma'
age = 38
has_android_phone = False
name, age, has_android_phone
person = {'Name': name, 'Age':age, 'HasAndroidPhone': has_android_phone}
print("{} is aged {}, and owns an {}.".format(
    person["Name"], 
    person["Age"], 
    "Android phone" if person["HasAndroidPhone"] else "iPhone"
))
# this is optional
for k in person.keys():
    print("The key {} has the value {} of the type{}.".format(k, person["Name"], type(k)))
jovian.commit(project=project_name,environment=None)
my_list = ["Pink", 0, False]
my_list
print('My favorite color is', my_list[0])
print('I have {} pet(s).'.format(my_list[1]))
if my_list[2]== True:
    print("I have previous programming experience")
else:
    print("I do not have previous programming experience")
my_list.append(5)
my_list
my_list.pop(0)
my_list
print("The list has {} elements.".format(my_list))
jovian.commit(project=project_name,environment=None)
# store the final answer in this variable
sum_of_numbers = 0

# perform the calculation here
for x in range(18,534):
    if (x%7)==0:
        sum_of_numbers += x
        
print(sum_of_numbers)
print('The sum of all the numbers divisible by 7 between 18 and 534 is', sum_of_numbers)
jovian.commit(project=project_name,environment=None)
cost_of_flying_plane = 5000
number_of_passengers = 29
price_of_ticket = 200
profit = cost_of_flying_plane - (number_of_passengers* price_of_ticket)
print('The company makes of a profit of {} dollars'.format(profit))
# this is optional
number_of_passengers = 29+12
cost_of_flying_plane = 5000*2

profit=cost_of_flying_plane - (number_of_passengers* price_of_ticket)

print(profit)
# this is optional
if profit>=0:
    print("The company makes an overall profit of {} dollars".format(profit))
else:
    print("The company makes an overall loss of {} dollars".format(profit))
jovian.commit(project=project_name,environment=None)
tweets = [
    "Wow, what a great day today!! #sunshine",
    "I feel sad about the things going on around us. #covid19",
    "I'm really excited to learn Python with @JovianML #zerotopandas",
    "This is a really nice song. #linkinpark",
    "The python programming language is useful for data science",
    "Why do bad things happen to me?",
    "Apple announces the release of the new iPhone 12. Fans are excited.",
    "Spent my day with family!! #happy",
    "Check out my blog post on common string operations in Python. #zerotopandas",
    "Freecodecamp has great coding tutorials. #skillup"
]

number_of_tweets = len(tweets)
print(number_of_tweets)

happy_words = ['great', 'excited', 'happy', 'nice', 'wonderful', 'amazing', 'good', 'best']
sad_words = ['sad', 'bad', 'tragic', 'unhappy', 'worst']
sample_tweet = tweets[0]
sample_tweet
is_tweet_happy = False

# Get a word from happy_words
for word in happy_words:
    # Check if the tweet contains the word
    if word in sample_tweet:
        # Word found! Mark the tweet as happy
        is_tweet_happy = True
        break
is_tweet_happy
# store the final answer in this variable
number_of_happy_tweets = 0

# perform the calculations here
for tweet in tweets:
    for word in happy_words:
        if word in tweet:
            number_of_happy_tweets +=1
            break
print("Number of happy tweets:", number_of_happy_tweets)
happy_fraction = number_of_happy_tweets/number_of_tweets 
print("The fraction of happy tweets is:", happy_fraction)
# store the final answer in this variable
number_of_sad_tweets = 0

# perform the calculations here
for tweet in tweets:
    for word in sad_words:
        if word in tweet:
            number_of_sad_tweets +=1
            break
print("Number of sad tweets:", number_of_sad_tweets)
sad_fraction = number_of_sad_tweets/number_of_tweets
print("The fraction of sad tweets is:", sad_fraction)
jovian.commit(project=project_name,environment=None)
sentiment_score = happy_fraction-sad_fraction
print("The sentiment score for the given tweets is", sentiment_score)
if sentiment_score>0:
    print("The overall sentiment is happy")
else:
    print("The overall sentiment is sad")
# store the final answer in this variable
number_of_neutral_tweets = 0

# perform the calculation here
number_of_neutral_tweets= number_of_tweets -(number_of_sad_tweets+number_of_happy_tweets)
    
neutral_fraction = number_of_neutral_tweets/number_of_tweets
print('The fraction of neutral tweets is', neutral_fraction)
jovian.commit(project=project_name,environment=None)