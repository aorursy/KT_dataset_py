
#used to read pdf
import fitz

#NLTK's word tokenizer
from nltk.tokenize import word_tokenize

#stopwords list of NLTK
from nltk.corpus import stopwords

########################### DEFINITIONS: ###########################
"""
1. porterConsonant(c/C): is a letter(c) other than A, E, I, O or U, and other than Y preceded by a consonant. A list ccc... of length greater than 0 will be denoted by C

2. porterVowel(v?V): a letter(v) that is not a consonant. A list vvv... of length greater than 0 will be denoted by V

3. M: equal to number of 'VC' in formstr 

4. Word form: the word form corresponding to [C](VC)^M[V] form

Some notations
*S - the stem ends with S (and similarly for the other letters).
*v* - the stem contains a vowel.
*d - the stem ends with a double consonant (e.g. -TT, -SS).
*o - the stem ends cvc, where the second c is not W, X or Y

"""
####################################################################

#definition of the class that implements porter algorithm
class myPorterStemmer:

	#function that returns a letter is a consonant or not
	def isCons(self, letter):
		consonantList = ['a', 'e', 'i', 'o', 'u']
		if letter in consonantList:
			return False
		else:
			return True

	#function that checks if the letter at ith position is a porterConsonant
	def isConsonant(self, word, i):
		letter = word[i]
		if self.isCons(letter):
			if letter == 'y' and self.isCons(word[i-1]):
				return False
			else:
				return True
		else:
			return False

	#function that checks if the letter is a vowel
	def isVowel(self, word, i):
		return not(self.isConsonant(word, i))

	
	#function that checks for *S
	def endsWith(self, stem, letter):
		if stem.endswith(letter):
			return True
		else:
			return False

	#function that checks for *v*
	def containsVowel(self, stem):
		for i in stem:
			if not self.isCons(i):
				return True
		return False

	#function that checks for *d
	def doubleCons(self, stem):
		if len(stem) >= 2:
			if self.isConsonant(stem, -1) and self.isConsonant(stem, -2):
				return True
			else:
				return False
		else:
			return False

	#function that checks for *o
	def cvc(self, word):
		if len(word) >= 3:
			f = -3
			s = -2
			t = -1
			third = word[t]
			if self.isConsonant(word, f) and self.isVowel(word, s) and self.isConsonant(word, t):
				if third != 'w' and third != 'x' and third != 'y':
					return True
				else:
					return False
			else:
				return False
		else:
			return False
			
	#function that returns the word form of the
	def getForm(self, word):
		
		form = []
		formStr = ''
		for i in range(len(word)):
			if self.isConsonant(word, i):
				if i != 0:
					prev = form[-1]
					if prev != 'C':
						form.append('C')
				else:
					form.append('C')
			else:
				if i != 0:
					prev = form[-1]
					if prev != 'V':
						form.append('V')
				else:
					form.append('V')
		for j in form:
			formStr += j
		return formStr

	#returns M value
	def getM(self, word):
		form = self.getForm(word)
		m = form.count('VC')
		return m

	
	#function checks if orig string ends with rem and replaces it with rep
	def replace(self, orig, rem, rep):
		result = orig.rfind(rem)
		base = orig[:result]
		replaced = base + rep
		return replaced

	#function checks if orig string has base string with M > 0 and ends with rem and replaces it with rep
	def replaceM0(self, orig, rem, rep):
		result = orig.rfind(rem)
		base = orig[:result]
		if self.getM(base) > 0:
			replaced = base + rep
			return replaced
		else:
			return orig

	#function checks if orig string has base string with M > 1 and ends with rem and replaces it with rep
	def replaceM1(self, orig, rem, rep):
		result = orig.rfind(rem)
		base = orig[:result]
		if self.getM(base) > 1:
			replaced = base + rep
			return replaced
		else:
			return orig
			
	#function that replaces 'sses' by 'ss', 'ies' by 'i', 'ss' by 'ss' and 's' by '' to get rid of plurals
	def step1a(self, word):
		if word.endswith('sses'):
			word = self.replace(word, 'sses', 'ss')
		elif word.endswith('ies'):
			word = self.replace(word, 'ies', 'i')
		elif word.endswith('ss'):
			word = self.replace(word, 'ss', 'ss')
		elif word.endswith('s'):
			word = self.replace(word, 's', '')
		else:
			pass
		return word

	""" function that checks if a word ends with 'eed','ed' or 'ing' and replaces these substrings with 'ee','' and ''. If after the replacements in case of 'ed' and 'ing', the resulting word
* ends with 'at','bl' or 'iz' : add 'e' to the end of the word
* ends with 2 consonants and its last letter isn't 'l','s' or 'z': remove last letter of the word
* has 1 as value of M and the cvc(word) returns true : add 'e' to the end of the word"""
	def step1b(self, word): 
		flag = False
		if word.endswith('eed'):
			result = word.rfind('eed')
			base = word[:result]
			if self.getM(base) > 0:
				word = base
				word += 'ee'
		elif word.endswith('ed'):
			result = word.rfind('ed')
			base = word[:result]
			if self.containsVowel(base):
				word = base
				flag = True
		elif word.endswith('ing'):
			result = word.rfind('ing')
			base = word[:result]
			if self.containsVowel(base):
				word = base
				flag = True
		if flag:
			if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
				word += 'e'
			elif self.doubleCons(word) and not self.endsWith(word, 'l') and not self.endsWith(word, 's') and not self.endsWith(word, 'z'):
				word = word[:-1]
			elif self.getM(word) == 1 and self.cvc(word):
				word += 'e'
			else:
				pass
		else:
			pass
		return word

	#function that replaces 'y' by 'i' In words ending with 'y' 
	def step1c(self, word):
		if word.endswith('y'):
			result = word.rfind('y')
			base = word[:result]
			if self.containsVowel(base):
				word = base
				word += 'i'
		return word

	#function that checks the value of M, and replaces the suffixes accordingly
	def step2(self, word):
		
		if word.endswith('ational'):
			word = self.replaceM0(word, 'ational', 'ate')
		elif word.endswith('tional'):
			word = self.replaceM0(word, 'tional', 'tion')
		elif word.endswith('enci'):
			word = self.replaceM0(word, 'enci', 'ence')
		elif word.endswith('anci'):
			word = self.replaceM0(word, 'anci', 'ance')
		elif word.endswith('izer'):
			word = self.replaceM0(word, 'izer', 'ize')
		elif word.endswith('abli'):
			word = self.replaceM0(word, 'abli', 'able')
		elif word.endswith('alli'):
			word = self.replaceM0(word, 'alli', 'al')
		elif word.endswith('entli'):
			word = self.replaceM0(word, 'entli', 'ent')
		elif word.endswith('eli'):
			word = self.replaceM0(word, 'eli', 'e')
		elif word.endswith('ousli'):
			word = self.replaceM0(word, 'ousli', 'ous')
		elif word.endswith('ization'):
			word = self.replaceM0(word, 'ization', 'ize')
		elif word.endswith('ation'):
			word = self.replaceM0(word, 'ation', 'ate')
		elif word.endswith('ator'):
			word = self.replaceM0(word, 'ator', 'ate')
		elif word.endswith('alism'):
			word = self.replaceM0(word, 'alism', 'al')
		elif word.endswith('iveness'):
			word = self.replaceM0(word, 'iveness', 'ive')
		elif word.endswith('fulness'):
			word = self.replaceM0(word, 'fulness', 'ful')
		elif word.endswith('ousness'):
			word = self.replaceM0(word, 'ousness', 'ous')
		elif word.endswith('aliti'):
			word = self.replaceM0(word, 'aliti', 'al')
		elif word.endswith('iviti'):
			word = self.replaceM0(word, 'iviti', 'ive')
		elif word.endswith('biliti'):
			word = self.replaceM0(word, 'biliti', 'ble')
		return word

	 #this function checks the value of M, and replaces the suffixes accordingly
	def step3(self, word):

		if word.endswith('icate'):
			word = self.replaceM0(word, 'icate', 'ic')
		elif word.endswith('ative'):
			word = self.replaceM0(word, 'ative', '')
		elif word.endswith('alize'):
			word = self.replaceM0(word, 'alize', 'al')
		elif word.endswith('iciti'):
			word = self.replaceM0(word, 'iciti', 'ic')
		elif word.endswith('ful'):
			word = self.replaceM0(word, 'ful', '')
		elif word.endswith('ness'):
			word = self.replaceM0(word, 'ness', '')
		return word

	#this function checks the value of M, and replaces the suffixes accordingly
	def step4(self, word):
		
		if word.endswith('al'):
			word = self.replaceM1(word, 'al', '')
		elif word.endswith('ance'):
			word = self.replaceM1(word, 'ance', '')
		elif word.endswith('ence'):
			word = self.replaceM1(word, 'ence', '')
		elif word.endswith('er'):
			word = self.replaceM1(word, 'er', '')
		elif word.endswith('ic'):
			word = self.replaceM1(word, 'ic', '')
		elif word.endswith('able'):
			word = self.replaceM1(word, 'able', '')
		elif word.endswith('ible'):
			word = self.replaceM1(word, 'ible', '')
		elif word.endswith('ant'):
			word = self.replaceM1(word, 'ant', '')
		elif word.endswith('ement'):
			word = self.replaceM1(word, 'ement', '')
		elif word.endswith('ment'):
			word = self.replaceM1(word, 'ment', '')
		elif word.endswith('ent'):
			word = self.replaceM1(word, 'ent', '')
		elif word.endswith('ou'):
			word = self.replaceM1(word, 'ou', '')
		elif word.endswith('ism'):
			word = self.replaceM1(word, 'ism', '')
		elif word.endswith('ate'):
			word = self.replaceM1(word, 'ate', '')
		elif word.endswith('iti'):
			word = self.replaceM1(word, 'iti', '')
		elif word.endswith('ous'):
			word = self.replaceM1(word, 'ous', '')
		elif word.endswith('ive'):
			word = self.replaceM1(word, 'ive', '')
		elif word.endswith('ize'):
			word = self.replaceM1(word, 'ize', '')
		elif word.endswith('ion'):
			result = word.rfind('ion')
			base = word[:result]
			if self.getM(base) > 1 and (self.endsWith(base, 's') or self.endsWith(base, 't')):
				word = base
			word = self.replaceM1(word, '', '')
		return word

	#function removes a final -e if m() > 1.
	def step5a(self, word):

		if word.endswith('e'):
			base = word[:-1]
			if self.getM(base) > 1:
				word = base
			elif self.getM(base) == 1 and not self.cvc(base):
				word = base
		return word

	#function changes -ll to -l if m() > 1
	def step5b(self, word):
		if self.getM(word) > 1 and self.doubleCons(word) and self.endsWith(word, 'l'):
			word = word[:-1]
		return word

	#function that executes porter algorithm steps
	def stem(self, word):
		word = self.step1a(word)
		word = self.step1b(word)
		word = self.step1c(word)
		word = self.step2(word)
		word = self.step3(word)
		word = self.step4(word)
		word = self.step5a(word)
		word = self.step5b(word)
		return word   

###Function that reads the pdf file passed as input and returns the file contents####
def readPDF(fileName):
	try:
		#reading from the pdf
		doc = fitz.open(fileName)
		
		#used to store the pdf contents
		pdfContent =[]
		
		#extracting text from each page in the pdf
		for page in doc:
			text=page.getText()
			pdfContent.append(text)
			
		doc.close()
		
		return pdfContent
		
	except Exception as e:
		print(e)

###Function that accepts a sentence as input and returns the sentence after stemming it using Porter Algorithm####		
def stemSentence(sentence):
	
	#create an object of class myPorterStemmer
	p = myPorterStemmer()
	
	#used to store the stemmed output
	ouputSentence=[]
	
	#the list of words returned by using NLTK's word tokenizer
	tokenWordList=word_tokenize(sentence)
	
	for word in tokenWordList:
		
		#finds the stem of the word
		ouputSentence.append(p.stem(word))
		ouputSentence.append(" ")
		
	return "".join(ouputSentence)
		
###Function that accepts a list of sentences as input and returns the list of sentences after stemming using Porter Algorithm####		
def stemDocument(docContent):
	
	try:
		#the list used to store the sentences after stemming
		stemmedContent=[]
		
		#list of stopwords as defined in NLTK
		stop_words = stopwords.words('english')
	
		#used to store a sentence after removing stop words
		sentence=""
		
		for page in docContent:
		
			for line in page.split("\n"):

				for word in line.split(" "):
										
					if (word not in stop_words):
						
						sentence+= word + " "
						
				if(sentence != "" and not(("\u2022" in sentence and len(sentence)==3))):
					#\u2022 is a bullet point in regex; bullet points alone are not treated as new sentences; instead they are combined with the text accompanting it
					stemmedContent.append(stemSentence(sentence))
					sentence=""
			
		return stemmedContent
	except Exception as e:
		print(e)

#main function
if __name__ == "__main__": 

	try:
	
		#reads the contents of assignment 1 pdf
		fileContents = readPDF('aparna_b160116cs.pdf')

		#applies the Porter Algorithm to stem the contents read from the pdf file
		stemmedOutput = stemDocument(fileContents)

		#writing the contents of the pdf after stemming to another file
		f = open("stemmedContents.txt", "w")
		
		for line in stemmedOutput:

			#writing each individual line from the pdf in a seperate line
			line += "\n"
			f.write(line)
			
		f.close()
	except Exception as e:
		print(e)
