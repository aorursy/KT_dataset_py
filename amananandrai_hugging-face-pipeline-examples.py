from transformers import pipeline
nlp = pipeline("sentiment-analysis")
result = nlp("I love trekking and yoga.")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
result = nlp("Racial discrimination should be outright boycotted.")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
nlp = pipeline("question-answering")
context = r"""

The property of being prime (or not) is called primality.

A simple but slow method of verifying the primality of a given number n is known as trial division.

It consists of testing whether n is a multiple of any integer between 2 and itself.

Algorithms much more efficient than trial division have been devised to test the primality of large numbers.

These include the Miller–Rabin primality test, which is fast but has a small probability of error, and the AKS primality test, which always produces the correct answer in polynomial time but is too slow to be practical.

Particularly fast methods are available for numbers of special forms, such as Mersenne numbers.

As of January 2016, the largest known prime number has 22,338,618 decimal digits.

"""
result = nlp(question="What is a simple method to verify primality?", context=context)
print(f"Answer: '{result['answer']}'")
result = nlp(question="As of January 2016 how many digits does the largest known prime consist of?", context=context)
print(f"Answer: '{result['answer']}'")
text_generator = pipeline("text-generation")
text= text_generator("A person must always work hard and", max_length=50, do_sample=False)[0]
print(text['generated_text'])
summarizer = pipeline("summarization")
ARTICLE = """The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972.

First conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space,

Apollo was later dedicated to President John F. Kennedy's national goal of "landing a man on the Moon and returning him safely to the Earth" by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress. 

Project Mercury was followed by the two-man Project Gemini (1962–66). 

The first manned flight of Apollo was in 1968.

Apollo ran from 1961 to 1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to 1966. 

Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions.

Apollo used Saturn family rockets as launch vehicles. 

Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973–74, and the Apollo–Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975.

 """
summary=summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)[0]
print(summary['summary_text'])
translator = pipeline("translation_en_to_de")
print(translator("A great obstacle to happiness is to expect too much happiness.", max_length=40)[0]['translation_text'])
