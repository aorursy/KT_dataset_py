import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load('en_core_web_sm')

args = input("Type your sentence here: ")
args = args.split(sep=', ')
analyzer = SentimentIntensityAnalyzer()
vs = analyzer.polarity_scores(args)
vsc = vs['compound']

if vsc > 0.50:
    print(f'\nSENTIMENT score: {vsc} -- VERY POSITIVE!')
elif vsc > 0.00 and vsc <= 0.50:
    print(f'\nSENTIMENT score: {vsc} -- Positive.')
elif vsc == 0.00:
    print(f'\nSENTIMENT score: {vsc} -- Neutral.')
elif vsc > -0.50 and vsc < 0.00:
    print(f'\nSENTIMENT score: {vsc} -- Negative.')
elif vsc > -1.0 and vsc < -0.50:
    print(f'\nSENTIMENT score: {vsc} -- VERY negative, dude.')
else:
    pass

doc = ' '.join(args)
document = nlp(doc)
print('\nPARTS OF SPEECH:\n')
for token in document:
    print(f'    {token.text:{15}} -->   {token.pos_:{10}}  {spacy.explain(token.tag_)}')

print()
print(f'ORIGINAL:  "{doc}"\n')
