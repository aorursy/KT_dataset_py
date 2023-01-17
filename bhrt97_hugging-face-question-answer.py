from transformers import pipeline

nlp = pipeline("sentiment-analysis")

print(nlp("I hate you"))
# print(nlp("I love you"))
print(nlp("""Life is beautiful, but not always. It has lots of problems you have to face everyday. Don't worry though! All these problems make you strong, it gives you courage to stand alone in future. Life is full of moments of joy, pleasure, success and comfort punctuated by misery, defeat, failures and problems. There is no human being on Earth, strong, powerful, wise or rich, who has not experienced, struggle, suffering or failure. You have to work hard to reach to the highest position. Life is full of paths, you just have to choose the right one. Life is interesting and amazing like the stars up in the skies.

With no doubt, Life is beautiful and full of celebrations. However you should always be ready to face adversity and challenges. There are difficult situations in life as well.Be careful!! You might get hurt too hard. Life is sometimes too selfish to think about yourself. Then life is too hard to handle. Falling in love! People tend to fall in love nowadays but i personally think the right time has to come... You might also get hurt in Love. You might be broken-hearted as the people say.

Life is the place where people treat everyone differently, racism exists as well as bullying. People tend to say bad stuff behind people's back. There are millions of people using horrible words to call people, People use people everyday. Life is not that easy in my view. Sometimes, all you want to do is sit alone and question yourself with hundred of questions"""))
nlp = pipeline("question-answering")

context = r"""
Life is beautiful, but not always. It has lots of problems you have to face everyday. Don't worry though! All these problems make you strong, it gives you courage to stand alone in future. Life is full of moments of joy, pleasure, success and comfort punctuated by misery, defeat, failures and problems. There is no human being on Earth, strong, powerful, wise or rich, who has not experienced, struggle, suffering or failure. You have to work hard to reach to the highest position. Life is full of paths, you just have to choose the right one. Life is interesting and amazing like the stars up in the skies.

With no doubt, Life is beautiful and full of celebrations. However you should always be ready to face adversity and challenges. There are difficult situations in life as well.Be careful!! You might get hurt too hard. Life is sometimes too selfish to think about yourself. Then life is too hard to handle. Falling in love! People tend to fall in love nowadays but i personally think the right time has to come... You might also get hurt in Love. You might be broken-hearted as the people say.

Life is the place where people treat everyone differently, racism exists as well as bullying. People tend to say bad stuff behind people's back. There are millions of people using horrible words to call people, People use people everyday. Life is not that easy in my view. Sometimes, all you want to do is sit alone and question yourself with hundred of questions"""

print(nlp(question="What is life?", context=context))
