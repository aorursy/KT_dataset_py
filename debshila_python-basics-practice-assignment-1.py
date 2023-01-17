# Install the library
!pip install jovian --upgrade --quiet
# Import it
import jovian
project_name='python-practice-assignment'
# Capture and upload a snapshot
jovian.commit(project=project_name, privacy='secret', evironment=None)
name = "Debshila"
age = 34
has_android_phone = True
name, age, has_android_phone
person = {"Name": name, "Age": age, "HasAndroidPhone":has_android_phone}
print("{} is aged {}, and owns an {}.".format(
    person["Name"], 
    person["Age"], 
    "Android phone" if person["HasAndroidPhone"] else "iPhone"
))
# this is optional
for key,value in person.items():
    print(f'The key {key} has the value {value} of the type {type(value)}')
jovian.commit(project=project_name,environment=None)
my_list = ['white', '0', True]
my_list
print('My favorite color is', my_list[0])
print('I have {} pet(s).'.format(my_list[1]))
if my_list[2]:
    print("I have previous programming experience")
else:
    print("I do not have previous programming experience")
my_list.append(0)
my_list
my_list.pop(0)
my_list
print("The list has {} elements.".format(len(my_list)))
jovian.commit(project=project_name,environment=None)
# store the final answer in this variable
# sum_of_numbers = 0

# perform the calculation here
div_by_7 = [i for i in range(18, 535) if i%7 == 0]
print(sum(div_by_7))
sum_of_numbers = sum(div_by_7)
print('The sum of all the numbers divisible by 7 between 18 and 534 is', sum_of_numbers)
jovian.commit(project=project_name,environment=None)
cost_of_flying_plane = 5000
number_of_passengers = 29
price_of_ticket = 200
revenue = (number_of_passengers * price_of_ticket)
print(revenue)
profit = revenue - cost_of_flying_plane
print('The company makes of a profit of {} dollars'.format(profit))
# this is optional
return_passengers = 12
return_profit = (return_passengers * price_of_ticket) - cost_of_flying_plane
print(return_profit)
# this is optional
if return_profit > 0:
    print("The company makes an overall profit of {} dollars".format(return_profit))
else:
    print("The company makes an overall loss of {} dollars".format(abs(return_profit)))
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
is_tweet_happy
# store the final answer in this variable
number_of_happy_tweets = 0

# perform the calculations here
for happy_word in happy_words:
    for tweet in tweets:
        if happy_word in tweet:
            number_of_happy_tweets +=1
    
        
print("Number of happy tweets:", number_of_happy_tweets)
happy_fraction = number_of_happy_tweets/number_of_tweets
print("The fraction of happy tweets is:", happy_fraction)
# store the final answer in this variable
number_of_sad_tweets = 0

# perform the calculations here
for sad_word in sad_words:
    for tweet in tweets:
        if sad_word in tweet:
            number_of_sad_tweets +=1
    

print("Number of sad tweets:", number_of_sad_tweets)
sad_fraction = number_of_sad_tweets/number_of_tweets
print("The fraction of sad tweets is:", sad_fraction)
jovian.commit(project=project_name,environment=None)
sentiment_score = happy_fraction - sad_fraction
print("The sentiment score for the given tweets is", sentiment_score)
if sentiment_score > 0:
    print("The overall sentiment is happy")
else:
    print("The overall sentiment is sad")
# store the final answer in this variable
number_of_neutral_tweets = 0

# perform the calculation here
number_of_neutral_tweets = number_of_tweets - (number_of_happy_tweets + number_of_sad_tweets)
neutral_fraction = number_of_neutral_tweets/number_of_tweets
print('The fraction of neutral tweets is', neutral_fraction)
jovian.commit(project=project_name,environment=None)
