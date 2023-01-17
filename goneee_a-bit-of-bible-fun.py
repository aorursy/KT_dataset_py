paragraph = (

    "There are 30 books of the Bible in this paragraph. Can you find them? This is a most remarkable puzzle. "

    "It was found by a gentleman in an airplane seat pocket, on a flight from Los Angeles to Honolulu, keeping "

    "him occupied for hours. He enjoyed it so much, he passed it on to some friends. One friend from Illinois "

    "worked on this while fishing from his john boat. Another friend studied it while playing his banjo. Elaine "

    "Taylor, a columnist friend, was so intrigued by it she mentioned it in her weekly newspaper column. Another "

    "friend judges the job of solving this puzzle so involving, she brews a cup of tea to help her nerves. There "

    "will be some names that are really easy to spot. That's a fact. Some people, however, will soon find themselves "

    "in a jam, especially since the book names are not necessarily capitalized. Truthfully, from answers we get, we "

    "are forced to admit it usually takes a minister or a scholar to see some of them at the worst. Research has "

    "shown that something in our genes is responsible for the difficulty we have in seeing the books in this "

    "paragraph. During a recent fund raising event, which featured this puzzle, the Alpha Delta Phi lemonade booth "

    "set a new record. The local paper, The Chronicle, surveyed over 200 patrons who reported that this puzzle was "

    "one of the most difficult they had ever seen. As Daniel Humana humbly puts it, 'The books are all right here in "

    "plain view hidden from sight.' Those able to find all of them will hear great lamentations from those who have to "

    "be shown. One revelation that may help is that books like Timothy and Samuel may occur without their numbers. "

    "Also, keep in mind, that punctuation and spaces in the middle are normal. A chipper attitude will help you "

    "compete really well against those who claim to know the answers. Remember, there is no need for a mad exodus; "

    "there really are 30 books of the Bible lurking somewhere in this paragraph waiting to be found")



book_names = ["Genesis",

"Exodus",

"Leviticus",

"Numbers",

"Deuteronomy",

"Joshua",

"Judges",

"Ruth",

"Samuel",

"Kings",

"Chronicles",

"Ezra",

"Nehemiah",

"Esther",

"Job",

"Psalm",

"Proverbs",

"Ecclesiastes",

"Song of Solomon",

"Isaiah",

"Jeremiah",

"Lamentations",

"Ezekiel",

"Daniel",

"Hosea",

"Joel",

"Amos",

"Obadiah",

"Jonah",

"Micah",

"Nahum",

"Habakkuk",

"Zephaniah",

"Haggai",

"Matthew",

"Mark",

"Luke",

"John",

"Acts",

"Romans",

"Corinthians",

"Galatians",

"Ephesians",

"Philippians",

"Colossians",

"Thessalonians",

"Timothy",

"Titus",

"Philemon",

"Hebrews",

"James",

"Peter",

"Jude",

"Revelation",

"Zechariah",

"Malachi"]
import re

def strip_non_letters(in_str):

    return re.sub(r'\W+', '', in_str)



len_str = len(paragraph)

for book in book_names:

    book_len = len(book)

    for idx, char in enumerate(paragraph):

        if char.lower() != book[0].lower():

            continue

        end_idx = idx + (2 * book_len)

        if end_idx > len_str:

            end_idx = len_str

        word_comparison = strip_non_letters(paragraph[idx:end_idx])

        if book.lower() in word_comparison[0:book_len].lower():

            print(book, idx)

            continue
import re

occurrences = []

books_not_found=[]

# Let's extract only letters after splitting the paragram into sentences

original_sentences = paragraph.split('.')

sentences = [re.sub(r'\W+', '', sentence.lower() ) for sentence in original_sentences]

for book in book_names: # for each book

    book = book.lower()

    book_occurrences=[] # temporarily record the sentence and position in the sentence

    for sentence_no,sentence in enumerate(sentences):

        while True: # We'll stop searching when the String.index function throws an exception i.e. it doesn't find the book

            try:

                # Retrieving the last index found in the sentence or 0 if first time search

                last_index= book_occurrences[-1]["position"] if len(book_occurrences) > 0 else 0

                book_occurrences.append({ # record the occurrence if found for later use

                    "sentence_no": sentence_no,

                    "sentence":original_sentences[sentence_no],

                    "book":book,

                    "position":sentence.index(book,last_index+1)

                })

            except ValueError:

                break

    if len(book_occurrences) > 0: # record that we found this book

        occurrences = occurrences+ book_occurrences

    else:

        books_not_found.append(book)

print("="*32)

print("%d Books Not Found in sentences" % len(books_not_found))

print("="*32)

print(", ".join(books_not_found))

print("="*32)

print("%d Books Found in sentences" % len(occurrences))

print("="*32)



print("For each sentence...")

occurrences.sort(key=lambda row : row["sentence_no"])

for find in occurrences:

    print("\t`%s` in sentence %d at approx %d characters from the start" % (

         find["book"][0].upper()+find["book"][1:],find["sentence_no"],find["position"]

    ))
import pandas as pd

occurrences.sort(key=lambda row : row["sentence_no"],reverse=True)

merged_data = pd.DataFrame(occurrences,index=range(1,len(occurrences)+1))

merged_data
import matplotlib.pyplot as plt

%matplotlib inline

merged_data.plot(x="sentence_no",kind="barh",figsize=(15,20))

plt.title("Distance from the beginning where each word was found")

plt.ylabel("Sentences")

plt.xlabel("Position In Sentence")

plt.style.use('fivethirtyeight')

for idx,find in enumerate(occurrences):

    plt.text(3,idx-0.1,find["book"].upper()+"  -   "+find["sentence"],size=10)