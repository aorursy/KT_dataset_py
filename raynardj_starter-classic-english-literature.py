import numpy as np

import pandas as pd

from sqlalchemy import create_engine as ce

from sqlalchemy import inspect

from pathlib import Path
DATA= Path("/kaggle/input/books.db")



engine = ce("sqlite:///"+str(DATA))

inspector = inspect(engine)
inspector.get_table_names()
books_df = pd.read_sql("books", con = engine)

books_df
author_df = pd.read_sql("authors", con = engine)

author_df
book_file_df = pd.read_sql("book_file", con = engine)

book_file_df
text_files_df = pd.read_sql("text_files", con = engine)

text_files_df
def searchAuthor(kw):

    return author_df[author_df.author.str.contains(kw)]



def searchBookByAuthor(kw):

    author_result = list(searchAuthor(kw).index)

    return books_df[books_df.author_id.isin(author_result)]
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import Column, Integer, String, Text, ForeignKey

from sqlalchemy.orm import relationship

Base = declarative_base()



def getSession(engine):

    from sqlalchemy.orm import sessionmaker

    return sessionmaker()(bind=engine)



class authorModel(Base):

    __tablename__ = 'authors'

    index = Column(Integer, primary_key = True)

    author = Column(Text)

    born = Column(Integer)

    death = Column(Integer)

    

    def __repr__(self):

        return "<Author: %s, %s to %s>"%(self.author,self.born_year,self.death_year)

    

    @property

    def born_year(self):

        if self.born < 9999: return self.born

        else: return "No Record"



    @property

    def death_year(self):

        if self.death < 9999: return self.death

        else: return "No Record"



class chapterModel(Base):

    __tablename__ = "text_files"

    index = Column(Integer, primary_key = True)

    fmt = Column(Text) # Format

    text = Column(Text) # Text content

    

    def __repr__(self):

        return "book file:%s"%(self.index)

    

class bookModel(Base):

    __tablename__ = "books"

    book_id = Column(Integer, primary_key = True)

    bookname = Column(Text)

    cate1 = Column(Text)

    author_id = Column(Integer,ForeignKey(authorModel.index))

    author = relationship(authorModel)

    

    def __repr__(self):

        return "<Book: %s>"%(self.bookname)

    

class bookChapterModel(Base):

    __tablename__ = "book_file"

    index = Column(Integer, primary_key = True)

    file_id = Column(Integer,ForeignKey(chapterModel.index))

    book_id = Column(Integer,ForeignKey(bookModel.book_id))

    file = relationship(chapterModel)

    book = relationship(bookModel)

    chapter = Column(Text())



    def __repr__(self):

        return "Book:%s with File:%s, Chapter:%s"%(self.book,self.file,self.chapter)

    

bookModel.maps = relationship(bookChapterModel)

bookModel.chapters = relationship(chapterModel, secondary = "book_file")

chapterModel.books = relationship(bookModel, secondary = "book_file")

chapterModel.maps = relationship(bookChapterModel)

authorModel.books = relationship(bookModel)
sess = getSession(engine)
sess.query(authorModel).all()
test_author = sess.query(authorModel).filter_by(index=64).first()

test_author
test_author.books[:5]
test_book = test_author.books[2]

test_book
test_book.chapters
print(test_book.chapters[0].text[:1000])