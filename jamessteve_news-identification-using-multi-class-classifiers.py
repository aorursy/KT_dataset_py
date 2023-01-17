import os

import matplotlib.pyplot as plt

import seaborn as sns



os.system("python -m pip install git+https://github.com/user1342/News-Article-Text-Classification.git")
from news_classification.news_topic_text_classifier import news_topic_text_classifier

my_predictor = news_topic_text_classifier()
my_predictor.get_all_categories()
fig = plt.figure(figsize=(8,6))

my_predictor._data_frame.groupby('category').body.count().plot.bar(ylim=0)

plt.show()
my_predictor.print_model_feature_data()
from sklearn.metrics import confusion_matrix

category_id_df = my_predictor._data_frame[['category', 'category_id']].drop_duplicates().sort_values('category_id')



conf_mat = confusion_matrix(my_predictor._y_test, my_predictor._y_pred)

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(conf_mat, annot=True, fmt='d',

            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print(my_predictor.get_category("When you’re watching your grocery budget, it’s handy to have a collection of great dinner recipes that cost $10 (or less) to make. It’s useful to know a few ways to make a meal from a few eggs and a can, how to turn an almost-empty peanut butter jar into sauce, and how to turn kale stems into a delicious dip instead of tossing them in the compost. But sometimes, no matter how great a recipe is, a meal just doesn’t feel complete without a side dish. After using Epi’s method for calculating recipe cost to price out a number of side dish recipes on this site, I learned that the most realistic way to keep side dishes affordable is to keep your ingredient list short. Depending on the season, your location, and other factors, your costs will vary. But luckily, a few ingredients are all you need to make a great side. Here are a few of our favorites around $5 dollars or less for four servings."))
print(my_predictor.get_category("The World Health Organization declared COVID-19 a pandemic 100 days ago on March 11. In a little more than three months, the coronavirus has infected more than 8.6 million people, and the death toll surpassed 458,000. The coronavirus, which causes the respiratory illness known as COVID-19, spread to nearly every continent, as doctors and nurses treat hundreds of patients per shift. Government officials scrambled to not only support their constituents, but also to implement ways to stem the rapidly spreading virus. Parts of the world plunged into unprecedented lockdowns, shuttering businesses and keeping people physically distant from one another, leading to economic decline. In the last few weeks, some restrictions have been lifted in a bid to restore normalcy in a pandemic-ridden world. But in light of reopening efforts, experts are concerned that the world could once again face the dark reality it faced at the early beginnings of the pandemic. In a matter of 100 days, the coronavirus has devastated populations around the world, and there doesn't yet seem to be an end in sight as scientists rush to develop a vaccine."))
print(my_predictor.get_category('Layla Moran says Sir Keir Starmer should be worried if she becomes the next leader of the Liberal Democrats, telling Business Insider that she wants to push the party to the political left and be "even more radical than Labour." Moran, the Member of Parliament for Oxford West and Abingdon, said she wanted to "fundamentally change how people perceive" the Liberal Democrats after the partys disappointing performance at the last general election. The party, which went into the December election fighting to stop Brexit, failed to capitalize on strong polling earlier in the year, ending up with one fewer seat in the House of Commons and then-leader Jo Swinson losing her seat. The Liberal Democrats will elect a new leader at the end of August, with Moran competing with acting leader Ed Davey and MP Wera Hobhouse to rebuild the party after it failed in its mission to keep Britain in the European Union. In an interview with Business Insider, Moran said she wanted to turn the Liberal Democrats into a "progressive, radical" force on the left of British politics, and put the legacy of the partys time in the Coalition government with David Camerons Conservatives fully behind it. "I want to take the Lib Dems back to our radical roots. I want us to be seen as more radical than Labour," she said.'))
print(my_predictor.get_category('European planemaker Airbus SE said on Friday it is extending furlough programs for 5,300 of its employees in Spain and the United Kingdom in its latest effort to cope with the impact of the coronavirus outbreak. "This will be effective from 20th May till 30th September and affects all Airbus Operations SL employees in Spain (with some exceptions), which means around 3,100 employees", an Airbus spokesman told Reuters in an emailed statement "Airbus Helicopters and Airbus DS employees in Spain are not impacted", the spokesman said. In the United Kingdom, the period of furlough for about 2,200 Airbus workers will start on July 20 and end on Aug. 9, the statement said. "In France, all employees of the commercial aircraft division are in some way affected", the spokesman said. He added about 29,500 employees in France were working on average about 30% shorter weeks. Sources told Reuters in May that Europes largest aerospace group was exploring restructuring plans involving the possibility of "deep" job cuts as it braced for a prolonged coronavirus crisis after furloughing thousands of workers.'))
my_predictor.create_data_set()
my_predictor.re_train()
my_predictor.print_model_feature_data()