from IPython.display import YouTubeVideo      

YouTubeVideo('qROvu7kjF-U')
import spacy

import pandas as pd
!python -m spacy download en_core_web_md
!python -m spacy link en_core_web_md enmd
!python -m spacy info enmd
nlp = spacy.load('enmd')
doc1= nlp("No SIM card was included. Finally got a SIM from carrier and the phone has to be power cycled numerous times to even make a call. Should have just went to carrier for the phone.")

doc2=nlp("Ordered for my in law. It was carrier locked to simple mobile but that’s what I was looking for anyways since we activated it with them. It came in it’s original Apple packaging with the shrink wrap and the SIM card attached to it. It works fine no problems")
print(doc1.similarity(doc2))
dc1=nlp('I like to eat apple daily for breakfast')

dc2=nlp('I like everything from Apple when it comes to phone')

print(dc1.similarity(dc2))
review1=nlp('The burger was small and did not have lettuce')

review2=nlp('The chicken was moist,juicy and spicy. It is worth spending some bucks ')



token1=nlp('eat')

reviews=[review1,review2]
[(i,token1,token1.similarity(i))for i in review1 if token1.similarity(i)>0.4] 
[(i,token1,token1.similarity(i))for i in review2 if token1.similarity(i)>0.3]
doc1= nlp("Sherry likes Donut")

token=nlp('waffle')[0]

print(doc1.similarity(token))
span = nlp("I like pizza and pasta")[2:5]

doc = nlp("McDonalds sells burgers")



print(span.similarity(doc))

vec_doc = nlp("The begger hit the jackpot")

# Access the vector via the token.vector attribute

print(vec_doc[1].vector)
tokens = nlp('Farsana.The weather is fantastic. We can go on road trip')

pd.DataFrame([(token.text, token.has_vector, token.vector_norm, token.is_oov)for token in tokens],columns=['Token','Vector?','Vector_norm','OOV?'])