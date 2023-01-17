# Generating speech using Hilary's email as Corpus
import pandas as pd

#Import email body text and convert to corpus
emaildf = pd.DataFrame.from_csv("../input/Emails.csv")
corpus = """r"""
for row in emaildf["ExtractedBodyText"]:
    corpus += str(row)
# modified from http://agiliq.com/blog/2009/06/generating-pseudo-random-text-with-markov-chains-u/

import random

class Markov(object):

	def __init__(self, corpus, chain_size=3):
		self.chain_size = chain_size
		self.cache = {}
		
		self.words = corpus.split()
		self.word_size = len(self.words)
		self.database()

	def words_at_position(self, i):
		"""Uses the chain size to find a list of the words at an index."""
		chain = []
		for chain_index in range(0, self.chain_size):
			chain.append(self.words[i + chain_index])
		return chain

	def chains(self):
		"""Generates chains from the given data string based on passed chain size.
		So if our string were:
			"What a lovely day"
		With a chain size of 3, we'd generate:
			(What, a, lovely)
		and
			(a, lovely, day)
		"""

		if len(self.words) < self.chain_size:
			return

		for i in range(len(self.words) - self.chain_size - 1):
			yield tuple(self.words_at_position(i))

	def database(self):
		for chain_set in self.chains():
			key = chain_set[:self.chain_size - 1]
			next_word = chain_set[-1]
			if key in self.cache:
				self.cache[key].append(next_word)
			else:
				self.cache[key] = [next_word]

	def generate_markov_text(self, size=25):
		seed = random.randint(0, self.word_size - 3)
		gen_words = []
		seed_words = self.words_at_position(seed)[:-1]
		gen_words.extend(seed_words)
		for i in range(size):
			last_word_len = self.chain_size - 1
			last_words = gen_words[-1 * last_word_len:]
			next_word = random.choice(self.cache[tuple(last_words)])
			gen_words.append(next_word)
		return ' '.join(gen_words)
    
markov = Markov(corpus)
#Just re-run this cell to get different results
email_length = 30
for i in range(10):
    print(str(i+1)+": ")
    print(markov.generate_markov_text(email_length))
