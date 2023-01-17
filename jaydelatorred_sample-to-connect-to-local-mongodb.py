from pymongo import MongoClient

from random import randint

from pprint import pprint
client = MongoClient(port=27017)
db=client.business

#Step 2: Create sample data

names = ['Kitchen','Animal','State', 'Tastey', 'Big','City','Fish', 'Pizza','Goat', 'Salty','Sandwich','Lazy', 'Fun']

company_type = ['LLC','Inc','Company','Corporation']

company_cuisine = ['Pizza', 'Bar Food', 'Fast Food', 'Italian', 'Mexican', 'American', 'Sushi Bar', 'Vegetarian']
for x in range(1, 501):

    business = {

        'name' : names[randint(0, (len(names)-1))] + ' ' + names[randint(0, (len(names)-1))]  + ' ' + company_type[randint(0, (len(company_type)-1))],

        'rating' : randint(1, 5),

        'cuisine' : company_cuisine[randint(0, (len(company_cuisine)-1))] 

    }

    #Step 3: Insert business object directly into MongoDB via isnert_one

    result=db.reviews.insert_one(business)
fivestar = db.reviews.find_one({'rating': 5})

fivestar
fivestarcount = db.reviews.count_documents({'rating': 5})

fivestarcount
db=client.business



ASingleReview = db.reviews.find_one({})

print('A sample document:')

pprint(ASingleReview)



result = db.reviews.update_one({'_id' : ASingleReview.get('_id') }, {'$inc': {'likes': 1}})

print('Number of documents modified : ' + str(result.modified_count))



UpdatedDocument = db.reviews.find_one({'_id':ASingleReview.get('_id')})

print('The updated document:')

pprint(UpdatedDocument)