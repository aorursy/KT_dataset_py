def sumAll(arr):

    myMin = min(arr)

    myMax = max(arr)

    return (myMax**2 + myMax)/2 - (myMin**2 - myMin)/2
sumAllCheck = [[1, 4], [4, 1], [5, 10], [10, 5]]

for check in sumAllCheck:

    print(sumAll(check))
def diffArray(arr1, arr2):

    return list(set(arr1) ^ set(arr2))
symDiffCheck = [([1, 2, 3, 5], [1, 2, 3, 4, 5]),

               (["diorite", "andesite", "grass", "dirt", "pink wool", "dead shrub"], ["diorite", "andesite", "grass", "dirt", "dead shrub"]),

               (["andesite", "grass", "dirt", "pink wool", "dead shrub"], ["diorite", "andesite", "grass", "dirt", "dead shrub"]),

               (["diorite", "andesite", "grass", "dirt", "dead shrub"], ["diorite", "andesite", "grass", "dirt", "dead shrub"]),

               ([1, "calf", 3, "piglet"], [1, "calf", 3, 4]),

               ([], ["snuffleupagus", "cookie monster", "elmo"]),

               ([1, "calf", 3, "piglet"], [7, "filly"])]

for check in symDiffCheck:

    print(diffArray(check[0], check[1]))
def destroyer(arr, *remove):

    return [x for x in arr if x not in remove]
print(destroyer([1, 2, 3, 1, 2, 3], 2, 3))

print(destroyer([1, 2, 3, 5, 1, 2, 3], 2, 3))

print(destroyer([3, 5, 1, 2, 2], 2, 3, 5))

print(destroyer([2, 3,2, 3], 2, 3))

print(destroyer(["tree", "hamburger", 53], "tree", 53))

print(destroyer(["possum", "trollo", 12, "safari", "hotdog", 92, 65, "grandma", "bugati", "trojan", "yacht"],

               "yacht", "possum", "trollo", "safari", "hotdog", "grandma", "bugati", "trojan"))
def filterFunc(myDict, sourceDict):

    for key in sourceDict.keys():

        if (key not in myDict) or (myDict[key] != sourceDict[key]):

            return False

    return True

def whatIsInAName(collection, source):

    return [item for item in collection if filterFunc(item, source)]
nameCheck = [([{"first": "Romeo", "last": "Montague"}, {"first": "Mercutio", "last": None}, {"first": "Tybalt", "last": "Capulet"}], {"last": "Capulet"}),

            ([{"apple": 1}, {"apple": 1, "bat": 2}, {"apple": 1}], {"apple": 1}),

            ([{ "apple": 1, "bat": 2 }, { "bat": 2 }, { "apple": 1, "bat": 2, "cookie": 2 }], { "apple": 1, "bat": 2 }),

            ([{ "apple": 1, "bat": 2 }, { "apple": 1 }, { "apple": 1, "bat": 2, "cookie": 2 }], { "apple": 1, "cookie": 2 }),

            ([{ "apple": 1, "bat": 2 }, { "apple": 1 }, { "apple": 1, "bat": 2, "cookie": 2 }, { "bat":2 }], { "apple": 1, "bat": 2 }),

            ([{"a": 1, "b": 2, "c": 3}], {"a": 1, "b": 9999, "c": 3})]

for check in nameCheck:

    print(whatIsInAName(check[0], check[1]))
import re

def spinalCase(s):

    withSpaces = re.sub('([a-z])([A-Z])', "\g<1> \g<2>", s)

    splitString = re.split("[^A-Za-z]", withSpaces)

    return "-".join(splitString).lower()
spinalCheck = ["This Is Spinal Tap", "thisIsSpinalTap", "The_Andy_Griffith_Show", "Teletubbies say Eh-oh", "AllThe-small Things"]

for check in spinalCheck:

    print(spinalCase(check))
import re

def translatePigLatin(s):

    cons_regex = "^[^aeiou]+"

    vow_regex = "[aeiou]"

    if (re.match(vow_regex, s) is not None):

        return s + "way"

    elif (re.search(vow_regex, s) is None):

        return s + "ay"

    else:

        tail = re.match(cons_regex, s).group(0)

        return s[len(tail):] + tail + "ay"
pigLatinTest = ["california", "paragraphs", "glove", "algorithm", "eight", "fly", "rhythm", "yolk"]

for test in pigLatinTest:

    print(test, translatePigLatin(test))
def myReplace(s, before, after):

    if (before[0].isupper()):

        cap_after = after[0].upper() + after[1:]

        return s.replace(before, cap_after)

    else:

        return s.replace(before, after)
replaceCheck = [("A quick brown fox jumped over the lazy dog", "jumped", "leaped"),

               ("Let us go to the store", "store", "mall"),

               ("He is Sleeping on the couch", "Sleeping", "sitting"),

               ("This has a spellngi error", "spellngi", "spelling"),

               ("His name is Tom", "Tom", "john"),

               ("Let us get back to more Coding", "Coding", "algorithms")]

for check in replaceCheck:

    print(check[0])

    print(myReplace(check[0], check[1], check[2]))
def pairElements(bases):

    pairDict = {"A": "T", "T": "A", "C": "G", "G": "C"}

    base_pairs = []

    for base in bases:

        base_pairs.append([base, pairDict[base]])

    return base_pairs
basesCheck = ["GCG", "ATCGA", "TTGAG", "CTCTA"]

for check in basesCheck:

    print(pairElements(check))
import string

def fearNotLetter(letterRange):

    ascii_lower = string.ascii_lowercase

    comparison_range = ascii_lower[ascii_lower.find(letterRange[0]):]

    for i in range(len(letterRange)):

        if letterRange[i] != comparison_range[i]:

            return comparison_range[i]

    return None
missing_letter_check = ["abce", "abcdefghjklmno", "stvwx", "bcdf", "abcdefghijklmnopqrstuvwxyz"]

for check in missing_letter_check:

    print(fearNotLetter(check))
def uniteUnique(*arrays):

    arrays_union = []

    for i in range(len(arrays)):

        for j in range(len(arrays[i])):

            if (arrays[i][j] not in arrays_union):

                arrays_union.append(arrays[i][j])

    return arrays_union
print(uniteUnique([1, 3, 2], [5, 2, 1, 4], [2, 1]))

print(uniteUnique([1, 3, 2], [1, [5]], [2, [4]]))

print(uniteUnique([1, 2, 3], [5, 2, 1]))

print(uniteUnique([1, 2, 3], [5, 2, 1, 4], [2, 1], [6, 7, 8]))
def convertHTML(string):

    html_dict = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;"}

    for key in html_dict.keys():

        string = string.replace(key, html_dict[key])

    return string
convertHTML_check = ["Dolce & Gabbana", "Hamburgers < Pizza < Tacos", "Sixty > twelve", 'Stuff in "quotation marks"', "Schindler's List", "<>", "abc"]

for check in convertHTML_check:

    print(check)

    print(convertHTML(check))
def sumFibs(num):

    alpha = (1 + 5**0.5)/2

    beta = (1 - 5**0.5)/2

    fib_sum = 0

    n = 1

    fib_n = 0

    while fib_n <= num:

        if fib_n % 2 != 0:

            fib_sum += fib_n

        fib_n = round((alpha**n - beta**n)/(alpha - beta))

        n += 1

    return fib_sum
fib_sum_check = [1, 1000, 4000000, 4, 75024, 75025]

for check in fib_sum_check:

    print(sumFibs(check))
def sumPrimes(num):

    primes = []

    for i in range(2, num + 1):

        divisibility_check = [(i/x).is_integer() for x in primes]

        if (True not in divisibility_check):

            primes.append(i)

    return sum(primes)
print(sumPrimes(10))

print(sumPrimes(977))
# Use the Euclidean algorithm to compute the GCD

def gcd(a, b):

    while b != 0:

        t = b

        b = a % b;

        a = t

    return a



def smallestCommons(arr):

    arr_min = min(arr)

    arr_max = max(arr)

    test_range = [i for i in range(arr_min, arr_max + 1)]

    # Set the initial value for lcm

    # Use the formula lcm(a, b) = abs(a*b)/gcd(a, b)

    lcm = abs(test_range[0]*test_range[1])/gcd(test_range[0], test_range[1])

    # Now compute the lcm of the entire sequence

    # Use the formula lcm(a, b, c) = lcm(lcm(a, b), c)

    for i in range(2, len(test_range)):

        lcm = abs(lcm*test_range[i])/gcd(lcm, test_range[i])

    return int(lcm)
lcm_check = [[1, 5], [5, 1], [2, 10], [1, 13], [23, 18]]

for check in lcm_check:

    print(smallestCommons(check))
def dropElements(arr, func):

    startIndex = len(arr) + 1

    for i in range(len(arr)):

        if (func(arr[i]) and (i < startIndex)):

            startIndex = i

    return arr[startIndex:]
print(dropElements([1, 2, 3, 4], lambda x: x >= 3))

print(dropElements([0, 1, 0, 1], lambda x: x == 1))

print(dropElements([1, 2, 3], lambda x: x > 0))

print(dropElements([1, 2, 3, 4], lambda x: x > 5))

print(dropElements([1, 2, 3, 7, 4], lambda x: x > 3))

print(dropElements([1, 2, 3, 9, 2], lambda x: x > 2))
def steamrollArray(arr):

    flattened_array = []

    # Loop through the elements of arr

    for i in range(len(arr)):

        if isinstance(arr[i], list):

            # If the element being checked is itself an array, recurse

            # Then concatenate the result to the flattened array

            flattened_array += steamrollArray(arr[i])

        else:

            # Otherwise, append the element to the flattened array

            flattened_array.append(arr[i])

    return flattened_array
steamroll_check = [[[["a"]], [["b"]]], [1, [2], [3, [[4]]]], [1, [], [3, [[4]]]], [1, {}, [3, [[4]]]]]

for check in steamroll_check:

    print(steamrollArray(check))
def binaryAgent(binary_string):

    # Split the binary string into individual numbers

    binary_letters = binary_string.split()

    # Process the binary letters

    # First convert each letter from a string into a base-10 integer using int(x, 2)

    # Then convert each number into it's corresponding character using

    # the built-in function chr(i)

    letters = [chr(int(x, 2)) for x in binary_letters]

    # Rejoin the characters using the String.join() function

    return "".join(letters)
print(binaryAgent("01000001 01110010 01100101 01101110 00100111 01110100 00100000 01100010 01101111 01101110 01100110 01101001 01110010 01100101 01110011 00100000 01100110 01110101 01101110 00100001 00111111"))

print(binaryAgent("01001001 00100000 01101100 01101111 01110110 01100101 00100000 01000110 01110010 01100101 01100101 01000011 01101111 01100100 01100101 01000011 01100001 01101101 01110000 00100001"))
def truthCheck(collection, pre):

    for item in collection:

        if (bool(item.get(pre)) == False):

            return False

    return True
print(truthCheck([{"user": "Tinky-Winky", "sex": "male"}, {"user": "Dipsy", "sex": "male"}, {"user": "Laa-Laa", "sex": "female"}, {"user": "Po", "sex": "female"}], "sex"))

print(truthCheck([{"user": "Tinky-Winky", "sex": "male"}, {"user": "Dipsy"}, {"user": "Laa-Laa", "sex": "female"}, {"user": "Po", "sex": "female"}], "sex"))

print(truthCheck([{"user": "Tinky-Winky", "sex": "male", "age": 0}, {"user": "Dipsy", "sex": "male", "age": 3}, {"user": "Laa-Laa", "sex": "female", "age": 5}, {"user": "Po", "sex": "female", "age": 4}], "age"))

print(truthCheck([{"name": "Pete", "onBoat": True}, {"name": "Repeat", "onBoat": True}, {"name": "FastFoward", "onBoat": None}], "onBoat"))

print(truthCheck([{"name": "Pete", "onBoat": True}, {"name": "Repeat", "onBoat": True, "alias": "Repete"}, {"name": "FastFoward", "onBoat": True}], "onBoat"))

print(truthCheck([{"single": "yes"}], "single"))

print(truthCheck([{"single": ""}, {"single": "double"}], "single"))

print(truthCheck([{"single": "double"}, {"single": None}], "single"))

print(truthCheck([{"single": "double"}, {"single": 0.0}], "single"))
def addTogether(*args):

    check_numerical = [isinstance(x, int) or isinstance(x, float) for x in args]

    if (not all(check_numerical)):

        return None

    if (len(args) == 2):

        return args[0] + args[1]

    if (len(args) == 1):

        return lambda x: addTogether(x, args[0])
print(addTogether(2, 3))

print(addTogether(2)(3))

print(addTogether("http://bit.ly/IqT6zt"))

print(addTogether(2, "3"))

print(addTogether(2)([3]))
class Person(object):

    def __init__(self, firstAndLast):

        """Assumes firstAndLast is a full name for the person"""

        self.name = firstAndLast.split()

    def getFirstName(self):

        return self.name[0]

    def getLastName(self):

        return self.name[1]

    def getFullName(self):

        return " ".join(self.name)

    def setFirstName(self, first):

        self.name[0] = first

    def setLastName(self, last):

        self.name[1] = last

    def setFullName(self, firstAndLast):

        self.name = firstAndLast.split()
bob = Person("Bob Ross")

print(bob.getFirstName())

print(bob.getLastName())

print(bob.getFullName())

bob.setFirstName("Haskell")

print(bob.getFullName())

bob.setLastName("Curry")

print(bob.getFullName())

bob.setFullName("Bobby Hill")

print(bob.getFirstName())

print(bob.getLastName())
import math

def orbitalPeriod(arr):

    GM = 398600.4418

    earthRadius = 6367.4447

    orbitalPeriods = []

    for elt in arr:

        elt_name = elt["name"]

        elt_alt = elt["avgAlt"]

        period = 2 * math.pi * math.sqrt((elt_alt + earthRadius)**3/GM)

        orbitalPeriods.append({"name": elt_name, "orbitalPeriod": round(period)})

    return orbitalPeriods
print(orbitalPeriod([{"name": "sputnik", "avgAlt": 35873.5553}]))

print(orbitalPeriod([{"name": "iss", "avgAlt": 413.6}, {"name": "hubble", "avgAlt": 556.7}, {"name": "moon", "avgAlt": 378632.553}]))