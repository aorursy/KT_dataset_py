%%capture

!pip install git+https://github.com/neuml/txtai
%%capture



from txtai.embeddings import Embeddings



# Create embeddings model, backed by sentence-transformers & transformers

embeddings = Embeddings({"method": "transformers", "path": "sentence-transformers/bert-base-nli-mean-tokens"})
import numpy as np



sections = ["US tops 5 million confirmed virus cases",

            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",

            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",

            "The National Park Service warns against sacrificing slower friends in a bear attack",

            "Maine man wins $1M from $25 lottery ticket",

            "Make huge profits without work, earn up to $100,000 a day"]



print("%-20s %s" % ("Query", "Best Match"))

print("-" * 50)



for query in ("feel good story", "climate change", "health", "war", "wildlife", "asia", "north america", "dishonest junk"):

    # Get index of best section that best matches query

    uid = np.argmax(embeddings.similarity(query, sections))



    print("%-20s %s" % (query, sections[uid]))
# Create an index for the list of sections

embeddings.index([(uid, text, None) for uid, text in enumerate(sections)])



print("%-20s %s" % ("Query", "Best Match"))

print("-" * 50)



# Run an embeddings search for each query

for query in ("feel good story", "climate change", "health", "war", "wildlife", "asia", "north america", "dishonest junk"):

    # Extract uid of first result

    # search result format: (uid, score)

    uid = embeddings.search(query, 1)[0][0]



    # Print section

    print("%-20s %s" % (query, sections[uid]))
embeddings.save("index")



embeddings = Embeddings()

embeddings.load("index")



uid = embeddings.search("climate change", 1)[0][0]

print(sections[uid])
!ls index