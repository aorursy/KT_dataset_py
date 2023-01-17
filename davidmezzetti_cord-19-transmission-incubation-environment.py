from IPython.display import display, Markdown



# Builds a markdown report

def report(file):

    display(Markdown(filename="../input/cord-19-analysis-with-sentence-embeddings/%s/%s.md" % (file, file)))



report("transmission")
