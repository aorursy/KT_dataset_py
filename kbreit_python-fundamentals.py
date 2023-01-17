name = "Jamie"

print(name)
people = ['Jamie', 'Kevin']

print(people)
print(people[0])
people = {'name': 'Jamie',

          'gender': 'male'}
print(people)

print(people['name'])
people = {'Jamie': {'gender': 'male'},

          'Kevin': {'gender': 'male'}

         }
print(people['Jamie']['gender'])
people = [{'name': 'Jamie',

           'gender': 'male'},

          {'name': 'Kevin',

           'gender': 'male'}

]
print(people[0])

print(people[0]['name'])
name = 'Jamie'

if name == 'Jamie':

    print("His name is Jamie")

elif name == 'Kevin':

    print("His name is Kevin")

else:

    print("His name isn't Jamie or Kevin")
people = ['Jamie', 'Kevin']

for person in people:

    print(person)
def print_name(name):

    print(name)
print_name("Jamie")
from random import randint



print(randint(0,10))



import random



print(random.randint(0,10))
admins = [{'name': 'Jamie',

           'email': 'jamie@google.com',

           'admin_id': 1},

          {'name': 'Kevin',

           'email': 'kevin@google.com',

           'admin_id': 2}

         ]

posts = [{'subject': 'Welcome to my Blog',

          'post_date': '1/1/2019',

          'admin_id': 1}

        ]



for post in posts:

    print(post['subject'])

    print(post['post_date'])

    id = post['admin_id']  # Get the admin_id for the post and store the value in a variable

    name = ""

    for admin in admins:  # Loop through admins to find the admin based on id

        if admin['admin_id'] == id:

            name = admin['name']

    print("By: " + name)