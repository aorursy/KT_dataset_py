from IPython.display import YouTubeVideo      
YouTubeVideo('FURNYtoLKvQ')
import re
def ngram(text,n):
    token = re.split("\\s+", text)
    ngrams=[]
    for i in range(len(token)-n+1):
        temp =[token[j]for j in range(i,i+n)]
        ngrams.append(" ".join(temp))
    return ngrams
text = "Movie is fantastically bad, still had a good time with my friends"
ngram(text,5)