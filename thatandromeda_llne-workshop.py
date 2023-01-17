# Pull in some tools we'll need.

import codecs

import glob

import gensim

%pylab inline
# Create a list of all of our book files.

book_filenames = sorted(glob.glob("../input/*.rtf"))

print("Found books:")

book_filenames
# Read each book into the book_corpus, doing some cleanup along the way.

book_corpus = []

for book_filename in book_filenames:

    with codecs.open(book_filename, "r", "utf-8") as book_file:

        book_corpus.append(

            gensim.models.doc2vec.TaggedDocument(

                gensim.utils.simple_preprocess( # Clean the text with simple_preprocess

                    book_file.read()),

                    ["{}".format(book_filename)])) # Tag each book with its filename
# Set up the model.

model = gensim.models.Doc2Vec(vector_size = 300, 

                              min_count = 3, 

                              epochs = 100)
model.build_vocab(book_corpus)

print("model's vocabulary length:", len(model.wv.vocab))
model.train(book_corpus,

            total_examples=model.corpus_count,

            epochs=model.epochs)
model.docvecs.most_similar(12) #The_Adventures_of_Tom_Sawyer_by_Mark_Twain
model.docvecs.most_similar(11) # The_Adventures_of_Sherlock_Holmes_by_Arthur_Conan_Doyle.rtf
model.docvecs.most_similar(16) # The_Prince_by_Nicolo_Machiavelli.rtf
model.wv.most_similar("monster")