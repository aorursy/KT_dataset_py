req = """faiss-gpu

"""



!echo {repr(req)} > requirements.txt

!cat ./requirements.txt
!mkdir dep

%cd dep

!pip download -r ../requirements.txt