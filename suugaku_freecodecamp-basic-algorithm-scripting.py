def convertToF(celsius):

    fahrenheit = (9/5) * celsius + 32

    return fahrenheit
celsiusTemps = [0, -30, 10, 20, 30]

for temp in celsiusTemps:

    print("Converting {0}C to Fahrenheit:".format(temp), convertToF(temp))
def reverseString(str):

    return str[::-1]
strings = ["hello", "Howdy", "Greetings from Earth"]

for string in strings:

    print("Reversed string:", reverseString(string))
def recursive_factorialize(num):

    if num == 0:

        return 1

    else:

        return num * recursive_factorialize(num - 1)

    

def iterative_factorialize(num):

    factorial = 1

    for i in range(1, num + 1):

        factorial *= i

    return factorial
factorials = [5, 10, 20, 0]

for num in factorials:

    print("Recursive factorial of", num, "=", recursive_factorialize(num))

    print("Iterative factorial of", num, "=", iterative_factorialize(num))
def findLongestWordLength(str):

    words = str.split()

    longest = 0

    for word in words:

        if len(word) > longest:

            longest = len(word)

    return longest
print(findLongestWordLength("The quick brown fox jumped over the lazy dog"))

print(findLongestWordLength("May the force be with you"))

print(findLongestWordLength("Google do a barrel roll"))

print(findLongestWordLength("What is the average airspeed velocity of an unladen swallow"))

print(findLongestWordLength("What if we try a super-long word such as otorhinolaryngology"))
def largestOfFour(arr):

    largest = []

    for ele in arr:

        largest.append(max(ele))

    return largest
print(largestOfFour([[4, 5, 1, 3], [13, 27, 18, 26], [32, 35, 37, 39], [1000, 1001, 857, 1]]))

print(largestOfFour([[4, 9, 1, 3], [13, 35, 18, 26], [32, 35, 97, 39], [1000000, 1001, 857, 1]]))

print(largestOfFour([[17, 23, 25, 12], [25, 7, 34, 48], [4, -10, 18, 21], [-72, -3, -17, -10]]))
def confirmEnding(s, target):

    targetLength = len(target)

    strEnding = s[len(s)-targetLength:]

    return strEnding == target
print(confirmEnding("Bastian", "n"))

print(confirmEnding("Congratulation", "on"))

print(confirmEnding("Connor", "n"))

print(confirmEnding("Walking on water and developing software from a specification are easy if both are frozen", "specification"))

print(confirmEnding("He has to give me a new name", "name"))

print(confirmEnding("Open sesame", "same"))

print(confirmEnding("Open sesame", "pen"))

print(confirmEnding("Open sesame", "game"))

print(confirmEnding("If you want to save our world, you must hurry. We dont know how much longer we can withstand the nothing", "mountain"))

print(confirmEnding("Abstraction", "action"))
def repeatStringNumTimes(s, num):

    if (num < 0):

        return ""

    else:

        return s*num
print(repeatStringNumTimes("*", 3))

print(repeatStringNumTimes("abc", 3))

print(repeatStringNumTimes("abc", 4))

print(repeatStringNumTimes("abc", 1))

print(repeatStringNumTimes("*", 8))

print(repeatStringNumTimes("abc", -2))
def truncateString(s, maxLength):

    if maxLength >= len(s):

        return s

    else:

        return s[0:maxLength] + "..."
print(truncateString("A-tisket a-tasket A green and yellow basket", 8))

print(truncateString("Peter Piper picked a peck of pickled peppers", 11))

print(truncateString("A-tisket a-tasket A green and yellow basket", len("A-tisket a-tasket A green and yellow basket")))

print(truncateString("A-tisket a-tasket A green and yellow basket", len("A-tisket a-tasket A green and yellow basket")+2))

print(truncateString("A-", 1))

print(truncateString("Absolutely Longer", 2))
def findElement(arr, func):

    for ele in arr:

        if func(ele):

            return ele
print(findElement([1, 3, 5, 8, 9, 10], lambda x: x%2 == 0))

print(findElement([1, 3, 5, 9], lambda x: x%2 == 0))
def booWho(boo):

    return type(boo) == bool
booWhoCheck = [True, False, [1, 2, 3], [1, 2, 3].copy, {"a":1}, 1, "a", "True", "False"]

for boo in booWhoCheck:

    print(booWho(boo))
def titleCase(string):

    words = string.split()

    titleCased = []

    for word in words:

        titleCased.append(word[0].upper() + word[1:].lower())

    return " ".join(titleCased)
titleCaseCheck = ["I'm a little tea pot", "sHoRt AnD sToUt", "HERE IS MY HANDLE AND HERE IS MY SPOUT"]

for phrase in titleCaseCheck:

    print(titleCase(phrase))
def frankenSplice(arr1, arr2, n):

    spliced = arr2[0:n]

    spliced.extend(arr1)

    spliced.extend(arr2[n:])

    return spliced
frankenSpliceCheck = [([1, 2, 3], [4, 5], 1), ([1, 2], ["a", "b"], 1), (["claw", "tentacle"], ["head", "shoulders", "knees", "toes"], 2)]

for check in frankenSpliceCheck:

    print(frankenSplice(check[0], check[1], check[2]))

    print("Checking for mutations:", check[0], check[1])
def bouncer(arr):

    return [item for item in arr if bool(item) == True]
bouncerCheck = [[7, "ate", "", False, 9], ["a", "b", "c"], [False, None, 0, (), ""],

               [1, None, 0j, 2, None]]

for check in bouncerCheck:

    print("Before bouncing:", check)

    print("After bouncing:", bouncer(check))
def getIndexToIns(arr, num):

    appended_arr = arr.copy()

    appended_arr.append(num)

    appended_arr.sort()

    return appended_arr.index(num)
indexCheck = [([10, 20, 30, 40, 50], 35), ([10, 20, 30, 40, 50], 30),

             ([40, 60], 50), ([3, 10, 5], 3), ([5, 3, 20, 3], 5),

             ([2, 20, 10], 19), ([2, 5, 10], 15), ([], 1)]

for check in indexCheck:

    print("Insertion Index:", getIndexToIns(check[0], check[1]))
def mutation(arr):

    str_1, str_2 = arr[0].lower(), arr[1].lower()

    for char in str_2:

        if char not in str_1:

            return False

    return True
mutationCheck = [["hello", "hey"], ["hello", "Hello"], ["zyxwvutsrqponmlkjihgfedcba", "qrstu"],

                ["Mary", "Army"], ["Mary", "Aarmy"], ["Alien", "line"], ["floor", "for"],

                ["hello", "neo"], ["voodoo", "no"]]

for check in mutationCheck:

    print(check, mutation(check))
def chunkArrayIntoGroups(arr, size):

    chunkedArray = []

    for i in range(0, len(arr), size):

        chunkedArray.append(arr[i:i + size])

    return chunkedArray
chunkCheck = [(["a", "b", "c", "d"], 2), ([0, 1, 2, 3, 4, 5], 3), ([0, 1, 2, 3, 4, 5], 2),

             ([0, 1, 2, 3, 4, 5], 4), ([0, 1, 2, 3, 4, 5, 6], 3), ([0, 1, 2, 3, 4, 5, 6, 7, 8], 4),

             ([0, 1, 2, 3, 4, 5, 6, 7, 8], 2)]

for check in chunkCheck:

    print(chunkArrayIntoGroups(check[0], check[1]))