!pip install chatterbot

!pip install --ignore-installed PyYAML
!pip install chatterbot

!pip install --ignore-installed PyYAML

!pip install chatterbot-corpus


"""from chatterbot import ChatBot



from chatterbot.trainers import ChatterBotCorpusTrainer



chatbot = ChatBot('Ron Obvious')



# Create a new trainer for the chatbot

trainer = ChatterBotCorpusTrainer(chatbot)



# Train the chatbot based on the english corpus

trainer.train("chatterbot.corpus.english")



# Get a response to an input statement

print(chatbot.get_response("Hello, how are you today?"))"""
from chatterbot import ChatBot

from chatterbot.trainers import ChatterBotCorpusTrainer



bot = ChatBot('Ron Obvious')



# Create a new trainer for the chatbot

trainer = ChatterBotCorpusTrainer(bot)



# Train the chatbot based on the english corpus

trainer.train("chatterbot.corpus.english")



while True:  

    message = input('You:')

    if message.strip() != 'Bye':

        reply = bot.get_response(message)

        print('ChatBot :',reply)



    if message.strip() == 'Bye':

        print('ChatBot : Bye')

        break