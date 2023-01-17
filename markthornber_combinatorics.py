import itertools
word = 'BREAKDOWN'
letterset = {letter for letter in word}
print(letterset)
data = itertools.combinations(letterset, 4)
subsets = set(data)
print(subsets)
numberOfSubsets = len(subsets)
print(numberOfSubsets)
vowels = {'A','E','I','O','U'}
myset = {'B','F','I','Z','A'}
vowelsInSet = myset.intersection(vowels)
print(vowelsInSet)
print(len(vowelsInSet))
set1 ={'A','E'}
set2 = {}
print(min(len(set1),1))
print(min(len(set2),1))
biglist = [min(len(set(choice).intersection(vowels)),1) for choice in subsets]
print(biglist)
numberWithVowels = sum(biglist)
print(numberWithVowels)
print(numberWithVowels,'/',numberOfSubsets)
def probabilityOfVowel(word,subsetlength):
    word = word.upper()
    letterset = {letter for letter in word}
    data = itertools.combinations(letterset, subsetlength)
    subsets = set(data)
    numberOfSubsets = len(subsets)
    vowels = {'A','E','I','O','U'}
    numberWithVowels = sum([min(len(set(choice).intersection(vowels)),1) for choice in subsets])
    answer = str(numberWithVowels)+'/'+str(numberOfSubsets)
    return answer
print(probabilityOfVowel('BREAKDOWN',4))
print(probabilityOfVowel('PythonJapes',5))