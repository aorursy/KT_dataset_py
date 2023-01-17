import gensim

# Set path to word embedings file

word2vecpath = "../input/GoogleNews-vectors-negative300.bin.gz"

word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vecpath, binary=True)
def get_gender_scores(sentence):
    # split sentence into words
    words = sentence.lower().split()
    
    # compute similarity for male and female direction.
    male_score = word2vec.n_similarity(['he', 'his', 'man', 'himself', 'son'], words)
    female_score = word2vec.n_similarity(['she', 'her', 'woman', 'herself', 'daughter'], words)
    
    return male_score, female_score
words = ['security', 'doctor', 'nurse', 'health insurance', 'english teacher', 'math', \
         'programmer', 'housekeeping', 'driver', 'plumber']
for word in words:
    male_score, female_score = get_gender_scores(word)
    stereotype = 'male' if male_score > female_score else 'female'
    print('"%s" is "%s" (male score: %f, female score: %f)' % (word, stereotype, male_score, female_score))
