class MyShisokuEnzanClass():

    

    #コンストラクタ　初期化メソッド　

    def __init__(self, a, b):

        self.valueA = a

        self.valueB = b



    #足し算メソッド    

    def tasu(self): 

        return str(self.valueA) + " たす " + str(self.valueB) + " は " + str(self.valueA + self.valueB)



    #引き算メソッド

    def hiku(self): 

        return str(self.valueA) + " ひく " + str(self.valueB) + " は " + str(self.valueA - self.valueB)



    #掛け算メソッド

    def kakeru(self): 

        return str(self.valueA) + " かける " + str(self.valueB) + " は " + str(self.valueA * self.valueB)

    

    #割り算メソッド

    def waru(self): 

        return str(self.valueA) + " わる " + str(self.valueB) + " は " + str(self.valueA / self.valueB)

    

myinstance = MyShisokuEnzanClass(5, 7)

print(myinstance.tasu())

print(myinstance.hiku())

print(myinstance.kakeru())

print(myinstance.waru())