
import gensim
model = gensim.models.Word2Vec.load('../input/embed.w2v')
#==========================================================
#Instructions: insert the symbols below. all symbols must be surrounded in quotes, delimited by comma
#arrPositive marks positive examples, arrNegative marks negative ones.
#press "play button on the upper left corner. results will enter the black section below.
arrPositive = ["appl","goog"]
arrNegative = "blabla"
#==========================================================
for word, sim in model.most_similar(positive=arrPositive, negative = arrNegative):
    print  (word + " " + str(sim) + '\n')




