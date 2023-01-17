from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import TabTokenizer
from nltk.tokenize import LineTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import SpaceTokenizer
text = '''Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago. In 1988, he enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. After graduating, he became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. He represented the 13th district for three terms in the Illinois Senate from 1997 until 2004, when he ran for the U.S. Senate. He received national attention in 2004 with his March primary win, his well-received July Democratic National Convention keynote address, and his landslide November election to the Senate. In 2008, he was nominated for president a year after his campaign began, after a close primary campaign against Hillary Clinton. He was elected over Republican John McCain and was inaugurated on January 20, 2009. Nine months later, he was named the 2009 Nobel Peace Prize laureate. Regarded as a centrist New Democrat, Obama signed many landmark bills into law during his first two years in office. The main reforms that were passed include the Patient Protection and Affordable Care Act (commonly referred to as the "Affordable Care Act" or "Obamacare"), the Dodd–Frank Wall Street Reform and Consumer Protection Act, and the Don't Ask, Don't Tell Repeal Act of 2010. The American Recovery and Reinvestment Act of 2009 and Tax Relief, Unemployment Insurance Reauthorization, and Job Creation Act of 2010 served as economic stimulus amidst the Great Recession. After a lengthy debate over the national debt limit, he signed the Budget Control and the American Taxpayer Relief Acts. In foreign policy, he increased U.S. troop levels in Afghanistan, reduced nuclear weapons with the United States–Russia New START treaty, and ended military involvement in the Iraq War. He ordered military involvement in Libya, contributing to the overthrow of Muammar Gaddafi. He also ordered the military operations that resulted in the deaths of Osama bin Laden and suspected Yemeni Al-Qaeda operative Anwar al-Awlaki.'''
print(sent_tokenize(text))
print(word_tokenize(text))
obj = LineTokenizer()
text = "This is a explanation of Line Tokenizer. \n Line Tokenizer is used to tokenize on lines."
print(obj.tokenize(text))
obj = SpaceTokenizer()
text = "This is a explanation of Line Tokenizer. \n Line Tokenizer is used to tokenize on lines."
print(obj.tokenize(text))
text = "This is a explanation \tof Line Tokenizer. \n Line Tokenizer is used to \ttokenize on lines."
obj = TabTokenizer()
print(obj.tokenize(text))
text = ":-) <> () {} [] :-p"
obj = TweetTokenizer()
print(obj.tokenize(text))