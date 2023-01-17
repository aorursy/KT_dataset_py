!nvidia-smi
!git clone https://github.com/huggingface/transformers.git

!pip install -U ./transformers

!pip install git+https://github.com/huggingface/nlp.git
import torch

import nlp

from transformers import LongformerTokenizerFast
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
import torch

from transformers import LongformerTokenizer, LongformerForQuestionAnswering



tokenizer = LongformerTokenizer.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # print ("device ",device)

# model = model.to(device)



text = "Huggingface has democratized NLP. Huge thanks to Huggingface for this."

question = "What has Huggingface done ?"

encoding = tokenizer.encode_plus(question, text, return_tensors="pt")

input_ids = encoding["input_ids"]



# default is local attention everywhere

# the forward method will automatically set global attention on question tokens

attention_mask = encoding["attention_mask"]



start_scores, end_scores = model(input_ids, attention_mask=attention_mask)

all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())



answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]

answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

# output => democratized NLP
import gc

gc.collect()
print(answer)
text = '''Essay Topic: Learning at Home during Lockdown:

My Parents and My Teachers



Ever since the lockdown started, I feel lonely at home.

I do have a brother but soon realized that talking to a person

or doing the same thing consistently can get monotonous.

Sometimes, I even feel that it would be better to go to school,

which a month-back I could not have thought of in a million

years.At my house, both my parents are doctors. Not that they do

not have holidays, they do! Somehow, the holidays do not

seem enough.My parents are treating COVID-19 patients and often discuss their healthcare. 

At times, I nd their conversations scary and mom calms me down by saying this will end soon.

Yet, I am hardly convinced with her explanations. In the little time that I get to talk to 

my friends, we discuss the current situation due to pandemic and its advantages, especially 

on the environment, as us human beings are in lockdown.

A few days ago, when my father and I were sitting in the balcony at night I looked up in the

sky and saw a lot more stars than I usually get to see. Even my mom told me that Yamuna

river is getting cleaner amidst the lockdown.

I also feel that my friends have their parents at home, spending quality time with them and

all having fun times, together. While they have fun, my parents are at the hospital treating

patients and, of course, this is something that makes me very proud. Still, it is not the same

as having them at home.

However, the advantage of not having parents at home is that I do not have to do any work

until they are back. A few weeks ago, I panicked thinking that I would not get to celebrate my

birthday on its due date, just as it was not celebrated the previous three consecutive years on

the birthday day, since my parents were busy treating patients of either typhoid, pneumonia

or dengue. A sigh of relief, this year it does not matter that much as long as my family and I

are safe.

I am also anxious about school; I hope that they do not take away our summer holidays to

make up for the missed school days. I always enjoyed attending Bharatanatyam dance

classes but now, due to the lockdown, we have these classes on Zoom, which I can only

imagine, must be hard for the teacher as she tries to make it look e
ortless. These classes,

on the other hand, do us some good, as we do not get to copy someone if we need to.

On weekdays the school gives us work, which I sometimes nd overwhelming, but it is

more work on their side, so that is impressive. Another thing I like is the kind of e
ort the

teachers are making to teach us by newer methods like making videos of concepts and

even dance steps, so hats o
 to them for that!

On days when we do have homework, my parents when home check it, which is good

because after the tiring day at work they still spend time with us.

Out of the many things I have learned during the lockdown, one main thing is that my parents

keep reminding through their example that we should keep hope and stay positive. 

'''
len(text.split())
def longformer(text,question):

    encoding = tokenizer.encode_plus(question, text, return_tensors="pt")

    input_ids = encoding["input_ids"]



    # default is local attention everywhere

    # the forward method will automatically set global attention on question tokens

    attention_mask = encoding["attention_mask"]



    start_scores, end_scores = model(input_ids, attention_mask=attention_mask)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())



    answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]

    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

    return answer
question = "What is my parents profession?"
longformer(text,question)
question = "What is one main thing I have learned during lockdown?"
longformer(text,question)
text = ''' The primary reasons for the American revolution were 

1. The Stamp Act 2. The Townshend Acts 3. The Boston Massacre 4. The Boston Tea Party 

5. The Coercive Acts 6. Lexington and Concord 7. British attacks on coastal towns'''

question = "What were the reasons  for American revolution ?"
longformer(text,question)
text = '''I spotted it in a junk shop in Bridport, a roll-top desk.

The man said it was early nineteenth century, and oak.

I had wanted one, but they were far too expensive. This

one was in a bad condition, the roll-top in several pieces,

one leg clumsily mended, scorch marks all down one

side. It was going for very little money. I thought I could

restore it. It would be a risk, a challenge, but I had to

have it. I paid the man and brought it back to my

workroom at the back of the garage. I began work on it

on Christmas Eve.

I removed the roll-top completely and pulled out the

drawers. The veneer had lifted almost everywhere — it looked like water damage to me. Both fire and water had

clearly taken their toll on this desk. The last drawer was

stuck fast. I tried all I could to ease it out gently. In the

end I used brute force. I struck it sharply with the side of

my fist and the drawer flew open to reveal a shallow space

underneath, a secret drawer. There was something in

there. I reached in and took out a small black tin box.

Sello-taped to the top of it was a piece of lined notepaper,

and written on it in shaky

handwriting: “Jim’s

last letter, received

January 25, 1915.

To be buried with

me when the

time comes.” I

knew as I did

it that it was

wrong of me to

open the box,

but curiosity

got the better of

my scruples. It

usually does.

Inside the box there was an envelope. The address

read: “Mrs Jim Macpherson, 12 Copper Beeches, Bridport,

Dorset.” I took out the letter and unfolded it. It was written

in pencil and dated at the top — “December 26, 1914

”.Dearest Connie,

I write to you in a much happier frame of mind because

something wonderful has just happened that I must tell you about at once. 

We were all standing to in our trenches

yesterday morning, Christmas morning. It was crisp and

quiet all about, as beautiful a morning as I’ve ever seen, as

cold and frosty as a Christmas morning should be.

I should like to be able to tell you that we began it.

But the truth, I’m ashamed to say, is that Fritz began it.

First someone saw a white flag waving from the trenches

opposite. Then they were calling out to us from across

no man’s land, “Happy Christmas, Tommy! Happy

Christmas!” When we had got over the surprise, some of

us shouted back, “Same to you, Fritz! Same to you!” I

thought that would be that. We all did. But then suddenly

one of them was up there in his grey greatcoat and waving

a white flag. “Don’t shoot, lads!” someone shouted. And

no one did. Then there was another Fritz up on the

parapet, and another. “Keep your heads down,” I told the

men, “it’s a trick.” But it wasn’t.

One of the Germans was waving a bottle above his

head. “It is Christmas Day, Tommy. We have schnapps.

We have sausage. We meet you? Yes?” By this time there

were dozens of them walking towards us across no man’s

land and not a rifle between them. Little Private Morris

was the first up. “Come on, boys. What are we waiting

for?” And then there was no stopping them. I was the

officer. I should have stopped them there and then, I

suppose, but the truth is that it never even occurred to

me I should. All along their line and ours I could see

men walking slowly towards one another, grey coats,

khaki coats meeting in the middle. And I was one of

them. I was part of this. In the middle of the war we

were making peace.You cannot imagine, dearest Connie, my feelings as

I looked into the eyes of the Fritz officer, who approached

me, hand outstretched. “Hans Wolf,” he said, gripping

my hand warmly and holding it. “I am from Dusseldorf.

I play the cello in the orchestra. Happy Christmas.”“Captain Jim Macpherson,” I replied. “And a Happy

Christmas to you too. I’m a school teacher from Dorset,

in the west of England.”

“Ah, Dorset,” he smiled. “I know this place. I know it

very well.” We shared my rum ration and his excellent

sausage. And we talked, Connie, how we talked. He spoke

almost perfect English. But it turned out that he had

never set foot in Dorset, never even been to England.

He had learned all he knew of England from school,

and from reading books in English. His favourite writer

was Thomas Hardy, his favourite book Far from the

Madding Crowd. So out there in no man’s land we talked

of Bathsheba and Gabriel Oak and Sergeant Troy and

Dorset. He had a wife and one son, born just six months

ago. As I looked about me there were huddles of khaki

and grey everywhere, all over no man’s land, smoking,

laughing, talking, drinking, eating. Hans Wolf and I

shared what was left of your wonderful Christmas cake,

Connie. He thought the marzipan was the best he had

ever tasted. I agreed. We agreed about everything, and

he was my enemy. There never was a Christmas party

like it, Connie.

Then someone, I don’t know who, brought out a

football. Greatcoats were dumped in piles to make

goalposts, and the next thing we knew it was Tommy

against Fritz out in the middle of no man’s land. Hans

Wolf and I looked on and cheered, clapping our hands

and stamping our feet, to keep out the cold as much as

anything. There was a moment when I noticed our

breaths mingling in the air between us. He saw it too

and smiled. “Jim Macpherson,” he said after a while,

“I think this is how we should resolve this war. A football

match. No one dies in a football match. No children are

orphaned. No wives become widows.”

“I’d prefer cricket,” I told him. “Then we Tommies

could be sure of winning, probably.” We laughed at

that, and together we watched the game. Sad to say,Connie, Fritz won, two goals to one. But as Hans Wolf

generously said, our goal was wider than theirs, so it

wasn’t quite fair.

The time came, and all too soon, when the game was

finished, the schnapps and the rum and the sausage

had long since run out, and we knew it was all over.

I wished Hans well and told him I hoped he would see

his family again soon, that the fighting would end and

we could all go home.

“I think that is what every soldier wants, on both

sides,” Hans Wolf said. “Take care, Jim Macpherson.

I shall never forget this moment, nor you.” He saluted

and walked away from me slowly, unwillingly, I felt.

He turned to wave just once and then became one of

the hundreds of grey-coated men drifting back towards

their trenches.

That night, back in our dugouts, we heard them

singing a carol, and singing it quite beautifully. It was

Stille Nacht, Silent Night. Our boys gave them a rousing

chorus of While Shepherds Watched. We exchanged

carols for a while and then we all fell silent. We had had

our time of peace and goodwill, a time I will treasure as

long as I live.Dearest Connie, by Christmas time next year, this

war will be nothing but a distant and terrible memory.

I know from all that happened today how much both

armies long for peace. We shall be together again soon,

I’m sure of it.

'''
len(text.split())
question = "Who had written the letter?"
longformer(text,question)
question = "Why was the letter written — what was the wonderful thing that had happened?"
longformer(text,question)
question = "What jobs did Hans Wolf and Jim Macpherson have when they were not soldiers?"
len(longformer(text,question).split())
context = text + '''I folded the letter again and slipped it carefully back

into its envelope. I kept awake all night. By morning I

knew what I had to do. I drove into Bridport, just a few

miles away. I asked a boy walking his dog where Copper

Beeches was. House number 12 turned out to be nothing

but a burned-out shell, the roof gaping, the windows

boarded-up. I knocked at the house next door and asked

if anyone knew the whereabouts of a Mrs Macpherson.

Oh yes, said the old man in his slippers, he knew her

well. A lovely old lady, he told me, a bit muddle-headed,

but at her age she was entitled to be, wasn’t she? A

hundred and one years old. She had been in the house

when it caught fire. No one really knew how the fire had

started, but it could well have been candles. She used

candles rather than electricity, because she always

thought electricity was too expensive. The fireman had

got her out just in time. She was in a nursing home

now, he told me, Burlington House, on the Dorchester

road, on the other side of town.I found Burlington House Nursing Home easily enough.

There were paper chains up in the hallway and a lighted

Christmas tree stood in the corner with a lopsided angel

on top. I said I was a friend come to visit Mrs Macpherson

to bring her a Christmas present. I could see through

into the dining room where everyone was wearing a paper

hat and singing. The matron had a hat on too and

seemed happy enough to see me. She even offered me a

mince pie. She walked me along the corridor.

“Mrs Macpherson is not in with the others,” she told

me. “She’s rather confused today so we thought it best

if she had a good rest. She has no family you know, no

one visits. So I’m sure she’ll be only too pleased to see

you.” She took me into a conservatory with wicker chairs

and potted plants all around and left me.

The old lady was sitting in a wheelchair, her hands

folded in her lap. She had silver white hair pinned into a

wispy bun. She was gazing out at

the garden. “Hello,” I said. She

turned and looked up at me

vacantly. “Happy Christmas,

Connie,” I went on. “I found

this. I think it’s yours.” As I was

speaking her eyes never left my

face. I opened the tin box and

gave it to her. That was the

moment her eyes lit up with

recognition and her face

became suffused with a sudden

glow of happiness. I explained

about the desk, about how I

had found it, but I don't think

she was listening. For a while she said nothing, but stroked the letter tenderly with her

fingertips.

Suddenly she reached out and took my hand. Her

eyes were filled with tears. “You told me you’d come home

by Christmas, dearest,” she said. “And here you are,

the best Christmas present in the world. Come closer,

Jim dear, sit down.”

I sat down beside her, and she kissed my cheek. “I

read your letter so often Jim, every day. I wanted to

hear your voice in my head. It always made me feel you

were with me. And now you are. Now you’re back you

can read it to me yourself. Would you do that for me,

Jim dear? I just want to hear your voice again. I’d love

that so much. And then perhaps we’ll have some tea.

I’ve made you a nice Christmas cake, marzipan all

around. I know how much you love marzipan.” '''
question = "Who did Connie Macpherson think her visitor was?"