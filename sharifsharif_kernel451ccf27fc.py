class StringManipulation:

    myWord = ""

    def get_string(self):

        print("Please Enter a Word : ")

        self.myWord = input()



    def print_string(self):

        print(self.myWord.upper())





myString = StringManipulation()

myString.get_string()

myString.print_string()