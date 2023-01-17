# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Code with Dictonary



import json

student = {

    "first_name": "Jake",

    "last_name": "Doyle"

}

json_data = json.dumps(student, indent=2)

print(json_data)

print(json.loads(json_data))
##Let us try to create an object 



## Error will be observed as we have not used serialized data

import json

class Student(object):

    def __init__(self, first_name: str, last_name: str):

        self.first_name = first_name

        self.last_name = last_name

        

student = Student(first_name="Jake", last_name="Doyle")

json_data = json.dumps(student)
## So we are using a __dict__ to get through the error



import json

class Student(object):

    def __init__(self, first_name: str, last_name: str):

        self.first_name = first_name

        self.last_name = last_name

        

student = Student(first_name="Jake", last_name="Doyle")

json_data = json.dumps(student.__dict__)

print(json_data)

print(Student(**json.loads(json_data)))
# or can be written without **#

d = json.loads(json_data)

Student(first_name=d["first_name"], last_name=d["last_name"])
##Seialization with more than one object



from typing import List

import json

class Student(object):

    def __init__(self, first_name: str, last_name: str):

        self.first_name = first_name

        self.last_name = last_name

class Team(object):

    def __init__(self, students: List[Student]):

        self.students = students

        

student1 = Student(first_name="Jake", last_name="Doyle")

student2 = Student(first_name="Jason", last_name="Durkin")

team = Team(students=[student1, student2])

json_data = json.dumps(team.__dict__, default=lambda o: o.__dict__, indent=4)

print(json_data)

#Deserialization



from typing import List

import json

class Student(object):

    def __init__(self, first_name: str, last_name: str):

        self.first_name = first_name

        self.last_name = last_name

class Team(object):

    def __init__(self, students: List[Student]):

        self.students = students

        

student1 = Student(first_name="Jake", last_name="Doyle")

student2 = Student(first_name="Jason", last_name="Durkin")

team = Team(students=[student1, student2])

json_data = json.dumps(team, default=lambda o: o.__dict__, indent=4)

print(json_data)

decoded_team = Team(**json.loads(json_data))

print(decoded_team)