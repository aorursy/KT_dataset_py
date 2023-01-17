PROBLEM_1 = 'https://www.kaggle.com https://www.google.com https://www.wikipedia.com'

PROBLEM_2 = '123, 012410, 01010, , 000, 111, 3495873, 3, not a number!, ...!@$,.'

PROBLEM_3 = 'Looking for many endings? You should only be seeing one match.'

PROBLEM_4 = 'Count the number of words in this sentence with at least five characters.'

PROBLEM_5 = 'Extract these two normally formatted phone numbers from this sentence: (123) 456 7890, 123-456-7890.'

PROBLEM_6 = '1234567890'

PROBLEM_7 = "An email address (imaginaryperson@imaginarymail.edu) in a sentence. Don't match Invalid_email@invalid."

PROBLEM_8 = "This is not a name, but Harry is. So is Susy. Sam should be missed as it's the first word in the sentence."

PROBLEM_9 = "https://www.kaggle.com https://www.google.com https://www.wikipedia.com http://phishing.com not.a.url gibberish41411 http https www.com"

PROBLEM_10 = "Weird whitespace   issues\t\t\t can be\n\n annoying."



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