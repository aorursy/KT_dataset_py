import re
PROBLEM_1 = 'https://www.kaggle.com https://www.google.com https://www.wikipedia.com'
pattern = re.compile(r"(https://)(www.)(\w+)(.com)")
matches = pattern.finditer(PROBLEM_1)
for match in matches:
    print(match.group(3))
PROBLEM_2 = '123, 012410, 01010, , 000, 111, 3495873, 3, not a number!, ...!@$,.'
pattern = re.compile(r"[2-9]+")
matches = pattern.findall(PROBLEM_2)
matches
PROBLEM_3 = 'Looking for many endings? You should only be seeing one match.'
pattern = re.compile(r"[A-Z][^\.!?]*[ing|ings][\.!?]")
matches = pattern.findall(PROBLEM_3)
len(matches)
PROBLEM_4 = 'Count the number of words in this sentence with at least five characters.'
pattern = re.compile(r"\b[\w+]{5,}\b")
matches = pattern.findall(PROBLEM_4)
for match in matches:
    print(match)
PROBLEM_5 = 'Extract these two normally formatted phone numbers from this sentence: (123) 456 7890, 123-456-7890.'
pattern = re.compile(r"\(?\d{3}\D+\d{3}\D\d{4}")
matches = pattern.findall(PROBLEM_5)
for match in matches:
    print(match)
PROBLEM_6 = '1234567890'
pattern = re.compile(r"\d{3}(\d{7})")
matches = pattern.findall(PROBLEM_6)
for match in matches:
    print(match)
PROBLEM_7 = "An email address (imaginaryperson@imaginarymail.edu) in a sentence. Don't match Invalid_email@invalid."
pattern = re.compile(r"\(\w+@(\w+)")
matches = pattern.findall(PROBLEM_7)
for match in matches:
    print(match)
PROBLEM_8 = "This is not a name, but Harry is. So is Susy. Sam should be missed as it's the first word in the sentence."
pattern = re.compile(r"(?<!^)(?<!\. )[A-Z][a-z]+")
matches = pattern.findall(PROBLEM_8)
for match in matches:
    print(match)
PROBLEM_9 = "https://www.kaggle.com https://www.google.com https://www.wikipedia.com http://phishing.com not.a.url gibberish41411 http https www.com"
pattern = re.compile(r"http://[a-zA-Z0-9]+\.\w+")
matches = pattern.findall(PROBLEM_9)
for match in matches:
    print(match)
PROBLEM_10 = "Weird whitespace           issues\t\t\t can be\n\n annoying."
re.sub("\s{2,}", " ", PROBLEM_10)
# AE 1
PHONE_FIELD_ENTRIES = '\n\n'.join([
    "1111111111",
    "222 222 2222",
    "333.333.3333",
    "(444) 444-4444",
    "Whitespace duplications can be hard to spot manually  555  555  5555 ",
    "Weird whitespace formats are still valid 666\t666\t6666",
    "Two separate phone numbers in one field 777.777.7777, 888 888 8888",
    "A common typo plus the US country code +1 999..999.9999",
    "Not a phone number, too many digits 1234567891011",
    "Not a phone number, too few digits 123.456",
    "Not a phone number, nine digits (123) 456-789",
                                   ])

pattern = re.compile(r"\(?\b\d{3}\)?[\s.]*\d{3}[\s.-]*\d{4}\b")
matches = pattern.findall(PHONE_FIELD_ENTRIES)
for match in matches:
    print(match)
import os
os.listdir("../input/")
import csv
pattern = re.compile(r"\d{4}\-\d{4}")
count = 0
with open("../input/documentation/documentation.csv", "r") as in_file: 
    csv_reader = csv.reader(in_file)
    for line in csv_reader:
        count += 1
        matches = pattern.findall(line[2])
        for match in matches:
            print("file_name: ", line[0], "\tYears: ", match)   
        if count == 100:
            break
count
# first 250 results
import re
import csv
count = 0
pattern = re.compile(r"([A-Z][a-z.-]+ (?:[A-Z][A-Za-z.] +?)?[A-Z][A-Za-z-']+)")
with open("../input/seattle-library-collection-inventory/library-collection-inventory.csv", "r") as in_file: 
    csv_reader = csv.reader(in_file)
    next(csv_reader)
    for line in csv_reader:
        string = line[1].split("/")
        if len(string) == 2:
            matches = pattern.findall(line[1].split("/")[1])
            for match in matches:
                print(match)
                count += 1
            if count == 250:
                break
count
