# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



class myClass:

    string = ""

    def __init__(self,string = "default"):

        self.string = string

    

    def setString(self, string):

        self.string = string

    

    def getString(self):

        return self.string

    

    def printString(self):

        print(self.string.upper())

        

print("what is your name? ")

name = input()

var = myClass(name)

print("nice to meet you " +var.getString())

print("How are you? ")

situation = input()

print("i'm happy that you are "+situation)
